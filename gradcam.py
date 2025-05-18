import argparse, os, cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ==== 1. 모델 정의 함수들 재사용 ====
from models.resnet50 import get_resnet50        
from models.densenet121 import get_densenet121  

# ==== 2. 단일 이미지 전처리 ====
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# ==== 3. Grad-CAM util ====
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.grads        = None
        # fwd hook
        target_layer.register_forward_hook(
            lambda _, __, output: setattr(self, "activations", output)
        )
        # bwd hook
        target_layer.register_full_backward_hook(
            lambda _, grad_in, grad_out: setattr(self, "grads", grad_out[0])
        )
        self.model.eval()

    @torch.no_grad()
    def _get_cam(self):
        # GAP over H,W → (B,C)
        weights = self.grads.mean(dim=(2, 3), keepdim=True)      # (B,C,1,1)
        cam     = (weights * self.activations).sum(dim=1)        # (B,H,W)
        cam     = F.relu(cam)
        # 정규화
        cam -= cam.min((1,2), keepdim=True)[0]
        cam /= cam.max((1,2), keepdim=True)[0] + 1e-8
        return cam  # 0~1

    def __call__(self, img_tensor, class_idx=None):
        img_tensor = img_tensor.unsqueeze(0)           # (1,3,H,W)
        img_tensor.requires_grad_()
        logits = self.model(img_tensor)                # fwd
        if class_idx is None:
            class_idx = logits.sigmoid().argmax().item()
        loss = logits[0, class_idx]                    # scalar
        self.model.zero_grad()
        loss.backward()                                # bwd
        cam = self._get_cam()[0].cpu().numpy()         # (H,W)
        return cam, class_idx


# ==== 4. 메인 ====
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 선택
    if args.arch == "resnet50":
        model = get_resnet50(num_classes=args.num_classes,
                             freeze_backbone=False)
        target_layer = model.layer4[-1]          # ResNet50 마지막 블록
    else:  # densenet121
        model = get_densenet121(num_classes=args.num_classes,
                                freeze_backbone=False)
        target_layer = model.features[-1]        # DenseNet 마지막 conv

    # 체크포인트 로드
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # 이미지 불러오기
    img_pil   = Image.open(args.image).convert("RGB")
    img_tensor= preprocess(img_pil).to(device)

    # Grad-CAM
    cam, cls = GradCAM(model, target_layer)(img_tensor, args.class_idx)

    # 시각화 → 원본에 overlay
    cam        = cv2.resize(cam, img_pil.size)            # (W,H)
    cam        = (cam*255).astype(np.uint8)
    heatmap    = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    result     = cv2.addWeighted(np.array(img_pil),
                                 0.5, heatmap, 0.5, 0)

    out_path = f"gradcam_{args.arch}_{os.path.basename(args.image)}"
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"[✓] saved → {out_path}  (class={cls})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="best_resnet50.pth 같은 체크포인트 경로")
    p.add_argument("--image", required=True,
                   help="Grad-CAM 뽑을 X-ray 이미지")
    p.add_argument("--arch", choices=["resnet50", "densenet121"],
                   default="resnet50")
    p.add_argument("--num_classes", type=int, default=14)  # NIH-14면 14
    p.add_argument("--class_idx", type=int, default=None,
                   help="보고싶은 클래스 idx (없으면 가장 높은 logit)")
    args = p.parse_args()
    main(args)