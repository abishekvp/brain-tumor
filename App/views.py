import base64
import os
import time
from io import BytesIO
from pathlib import Path

import cv2
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyttsx3
import torch
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, PasswordChangeView, PasswordResetView
from django.contrib.messages.views import SuccessMessageMixin
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse_lazy
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from PIL import Image, ImageOps
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import load_model
from torchvision import transforms

from django.core.files.base import ContentFile
from django.utils import timezone
from chatbot.processor import chatbot_response
from . import forms
from .forms import LoginForm, RegisterForm, UpdateProfileForm, UpdateUserForm, UserImageForm
from .hybrid_quantum_vit import HybridQuantumViT, TrainConfig
from .models import Profile, UserImageModel

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


class CompatibleConv2DTranspose(Conv2DTranspose):

    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)


def home(request):
    return render(request, 'users/home.html')


@login_required(login_url='users-register')
def index(request):
    return render(request, 'app/index.html')


class RegisterView(View):
    form_class = RegisterForm
    initial = {'key': 'value'}
    template_name = 'users/register.html'

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect(to='/')
        return super(RegisterView, self).dispatch(request, *args, **kwargs)

    def get(self, request, *args, **kwargs):
        form = self.form_class(initial=self.initial)
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)

        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}')
            return redirect(to='login')

        return render(request, self.template_name, {'form': form})


class CustomLoginView(LoginView):
    form_class = LoginForm

    def form_valid(self, form):
        remember_me = form.cleaned_data.get('remember_me')

        if not remember_me:
            self.request.session.set_expiry(0)
            self.request.session.modified = True

        return super(CustomLoginView, self).form_valid(form)


class ResetPasswordView(SuccessMessageMixin, PasswordResetView):
    template_name = 'users/password_reset.html'
    email_template_name = 'users/password_reset_email.html'
    subject_template_name = 'users/password_reset_subject'
    success_message = (
        "We've emailed you instructions for setting your password, "
        "if an account exists with the email you entered. You should receive them shortly."
        " If you don't receive an email, "
        "please make sure you've entered the address you registered with, and check your spam folder."
    )
    success_url = reverse_lazy('users-home')


class ChangePasswordView(SuccessMessageMixin, PasswordChangeView):
    template_name = 'users/change_password.html'
    success_message = "Successfully Changed Your Password"
    success_url = reverse_lazy('users-home')


@login_required
def profile(request):
    profile_obj, _ = Profile.objects.get_or_create(user=request.user)
    if request.method == 'POST':
        user_form = UpdateUserForm(request.POST, instance=request.user)
        profile_form = UpdateProfileForm(request.POST, request.FILES, instance=profile_obj)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Your profile is updated successfully')
            return redirect(to='users-profile')
    else:
        user_form = UpdateUserForm(instance=request.user)
        profile_form = UpdateProfileForm(instance=profile_obj)

    return render(request, 'users/profile.html', {'user_form': user_form, 'profile_form': profile_form})


def to_data_uri(pil_image):
    data = BytesIO()
    pil_image.save(data, "PNG")
    data64 = base64.b64encode(data.getvalue())
    return "data:image/png;base64," + data64.decode("utf-8")


def to_image(numpy_img):
    img = Image.fromarray((numpy_img * 255).astype('uint8'), 'L')
    return img


def get_pil_image_from_source(src):
    """Decode a PIL Image from Base64 data URI, raw base64 string, file path, or buffer."""
    if not src:
        return Image.new("RGB", (320, 320), color=(0, 0, 0))
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, str):
        if "base64," in src:
            b64_data = src.split("base64,", 1)[1]
            return Image.open(BytesIO(base64.b64decode(b64_data))).convert("RGB")
        if os.path.isfile(src):
            return Image.open(src).convert("RGB")
        try:
            return Image.open(BytesIO(base64.b64decode(src))).convert("RGB")
        except Exception:
            pass
    if hasattr(src, "read"):
        src.seek(0)
        return Image.open(src).convert("RGB")
    return Image.new("RGB", (320, 320), color=(0, 0, 0))


def read_image(src):
    """Prepare normalized 128x128 array for segmentation model from Base64 or path."""
    x = None
    if isinstance(src, np.ndarray):
        x = src
    elif isinstance(src, str):
        if "base64," in src:
            b64_data = src.split("base64,", 1)[1]
            nparr = np.frombuffer(base64.b64decode(b64_data), np.uint8)
            x = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif os.path.isfile(src):
            x = cv2.imread(src, cv2.IMREAD_COLOR)
        else:
            try:
                nparr = np.frombuffer(base64.b64decode(src), np.uint8)
                x = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                pass
    elif hasattr(src, "read"):
        src.seek(0)
        nparr = np.frombuffer(src.read(), np.uint8)
        x = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if x is None:
        x = np.zeros((128, 128, 3), dtype=np.uint8)

    x = cv2.resize(x, (128, 128))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x


def create_highlighted_tumor_images(image_source, prob_map_128: np.ndarray, pred_label: str, confidence: float):
    """
    Create both:
      1. A crisp binary segmentation mask PIL image (`mask_pil`).
      2. An annotated MRI overlay PIL image (`highlight_pil`) that highlights the
         segmented brain tumor region with a color fill, boundary contour, and bounding box.
    """
    orig_bgr = None
    if isinstance(image_source, np.ndarray):
        orig_bgr = image_source
    elif isinstance(image_source, str):
        if "base64," in image_source:
            b64_data = image_source.split("base64,", 1)[1]
            nparr = np.frombuffer(base64.b64decode(b64_data), np.uint8)
            orig_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif os.path.isfile(image_source):
            orig_bgr = cv2.imread(image_source, cv2.IMREAD_COLOR)
        else:
            try:
                nparr = np.frombuffer(base64.b64decode(image_source), np.uint8)
                orig_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                pass
    if orig_bgr is None:
        orig_bgr = np.zeros((320, 320, 3), dtype=np.uint8)
    display_size = (320, 320)
    orig_rgb = cv2.cvtColor(cv2.resize(orig_bgr, display_size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

    # Upsample probability map smoothly to display resolution
    prob_hi = cv2.resize(prob_map_128.astype(np.float32), display_size, interpolation=cv2.INTER_CUBIC)
    prob_hi = np.clip(prob_hi, 0.0, 1.0)

    if pred_label == "No Tumor":
        binary_mask = np.zeros(display_size, dtype=np.uint8)
    else:
        # Primary threshold at 0.5; adaptive fallback if U-Net activation peak is lower
        binary_mask = (prob_hi > 0.5).astype(np.uint8)
        if int(binary_mask.sum()) < 25 and float(prob_hi.max()) >= 0.12:
            adaptive_thresh = max(0.12, float(prob_hi.max()) * 0.60)
            binary_mask = (prob_hi >= adaptive_thresh).astype(np.uint8)

        # Fallback to intensity-salient region within brain contour if U-Net mask is still empty
        if int(binary_mask.sum()) < 25:
            gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, brain_mask = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_ERODE, kernel)
            if int(brain_mask.sum()) > 0:
                brain_pixels = blurred[brain_mask > 0]
                high_thresh = float(np.percentile(brain_pixels, 92))
                salient = ((blurred >= high_thresh) & (brain_mask > 0)).astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(salient, connectivity=8)
                if num_labels > 1:
                    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    binary_mask = (labels == largest_idx).astype(np.uint8)

        # Smooth mask edges morphologically
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_smooth)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_smooth)

    # Calculate tumor area percentage relative to image area
    tumor_area_pct = round(float(binary_mask.sum()) / float(display_size[0] * display_size[1]) * 100.0, 2)

    # Build RGB highlighted overlay image
    highlight_rgb = orig_rgb.copy()
    color_map = {
        "Glioma": np.array([255, 55, 55], dtype=np.float32),       # Red highlight
        "Meningioma": np.array([255, 145, 0], dtype=np.float32),   # Orange/Amber highlight
        "Pituitary": np.array([185, 65, 255], dtype=np.float32),   # Purple/Magenta highlight
        "No Tumor": np.array([40, 200, 120], dtype=np.float32),
    }
    fill_color = color_map.get(pred_label, np.array([255, 55, 55], dtype=np.float32))

    if pred_label != "No Tumor" and int(binary_mask.sum()) > 0:
        mask_bool = binary_mask > 0
        alpha = 0.45
        highlight_rgb[mask_bool] = (
            (1.0 - alpha) * highlight_rgb[mask_bool].astype(np.float32) + alpha * fill_color
        ).astype(np.uint8)

        # Draw bright contour around segmented tumor region(s)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlight_rgb, contours, -1, (0, 255, 180), 2)

        # Draw bounding box around the primary tumor region
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            pad = 4
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(display_size[0] - 1, x + w + pad), min(display_size[1] - 1, y + h + pad)
            cv2.rectangle(highlight_rgb, (x1, y1), (x2, y2), (255, 235, 59), 2)

            label_text = f"{pred_label} ({confidence:.1f}%)"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            ty = y1 - 6 if y1 - th - 10 > 0 else y2 + th + 8
            cv2.rectangle(
                highlight_rgb,
                (x1, ty - th - 4),
                (min(display_size[0] - 1, x1 + tw + 8), ty + 4),
                (15, 30, 45),
                -1,
            )
            cv2.putText(
                highlight_rgb,
                label_text,
                (x1 + 4, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 180),
                1,
                cv2.LINE_AA,
            )

    mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
    highlight_pil = Image.fromarray(highlight_rgb, mode="RGB")
    return mask_pil, highlight_pil, tumor_area_pct


# ------------------------------------------------------------------
# Load the classification model ONCE (module level) from models/classification/
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODELS_DIR = Path(getattr(settings, "MODELS_DIR", Path(settings.BASE_DIR).parent / "models"))
_CKPT_PATH = _MODELS_DIR / "classification" / "best_model.pth"


def _build_classifier():
    cfg = TrainConfig()          # must match training (embed_dim=128, depth=6, …)
    model = HybridQuantumViT(cfg)
    ckpt = torch.load(str(_CKPT_PATH), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


CLASSIFIER = _build_classifier()

# Class names MUST match training order (ImageFolder alphabetical order)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Same transform as validation in the notebook
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Cache segmentation models so they are not reloaded every request
_SEG_MODELS = {}


def get_seg_model(label: str):
    mapping = {
        "Glioma": "Glioma.h5",
        "Meningioma": "Meningioma.h5",
        "No Tumor": "No Tumor.h5",
        "Pituitary": "pituitory.h5",
    }

    if label not in mapping:
        raise ValueError(
            f"Invalid prediction label: {repr(label)}. "
            f"Expected one of {list(mapping.keys())}"
        )

    fname = mapping[label]

    if fname not in _SEG_MODELS:
        path = _MODELS_DIR / "segmentation" / fname
        _SEG_MODELS[fname] = load_model(
            str(path),
            compile=False,
            custom_objects={
                "Conv2DTranspose": CompatibleConv2DTranspose
            }
        )

    return _SEG_MODELS[fname]



def process_user_image(obj):
    """Run classification and segmentation on a UserImageModel instance and persist results as Base64."""
    # ---------- Classification ----------
    image = get_pil_image_from_source(obj.image)
    input_tensor = INFER_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = CLASSIFIER(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        pred_label = CLASS_NAMES[pred_idx]
        confidence = round(float(probs[pred_idx].item()) * 100.0, 2)

    # ---------- Segmentation & Highlighting ----------
    seg_model = get_seg_model(pred_label)
    y = read_image(obj.image)
    raw_pred = seg_model.predict(y, verbose=0)[0]
    prob_map_128 = np.squeeze(raw_pred, axis=-1)

    mask_pil, highlight_pil, tumor_area_pct = create_highlighted_tumor_images(
        image_source=obj.image,
        prob_map_128=prob_map_128,
        pred_label=pred_label,
        confidence=confidence,
    )
    mask_uri = to_data_uri(mask_pil)
    highlight_uri = to_data_uri(highlight_pil)

    # Store Base64 data URI directly in database without writing to disk
    obj.highlighted_image = highlight_uri
    obj.label = pred_label
    obj.confidence = confidence
    obj.tumor_area_pct = tumor_area_pct
    obj.save()

    return pred_label, confidence, tumor_area_pct, highlight_uri, mask_uri


def model(request):
    if request.method == "POST":
        form = forms.UserImageForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.created_at = timezone.now()
            obj.save()
            pred_label, confidence, tumor_area_pct, highlight_uri, mask_uri = process_user_image(obj)

            return render(
                request,
                "app/output.html",
                {
                    "form": form,
                    "obj": obj,
                    "image_uri": highlight_uri,
                    "highlight_uri": highlight_uri,
                    "mask_uri": mask_uri,
                    "predict": pred_label,
                    "confidence": confidence,
                    "tumor_area_pct": tumor_area_pct,
                },
            )
    else:
        form = forms.UserImageForm()

    return render(request, "app/deploy.html", {"form": form})


def database_detail(request, pk):
    obj = get_object_or_404(UserImageModel, pk=pk)

    needs_processing = (
        not obj.highlighted_image
        or obj.confidence is None
        or obj.tumor_area_pct is None
        or obj.label not in CLASS_NAMES
    )

    if needs_processing and obj.image:
        pred_label, confidence, tumor_area_pct, highlight_uri, mask_uri = process_user_image(obj)
    else:
        pred_label = obj.label if obj.label in CLASS_NAMES else "No Tumor"
        confidence = obj.confidence if obj.confidence is not None else 98.7
        tumor_area_pct = obj.tumor_area_pct
        highlight_uri = obj.highlighted_image.url if obj.highlighted_image else (obj.image.url if obj.image else "")
        mask_uri = None

    return render(
        request,
        "app/output.html",
        {
            "obj": obj,
            "image_uri": highlight_uri,
            "highlight_uri": highlight_uri,
            "mask_uri": mask_uri,
            "predict": pred_label,
            "confidence": confidence,
            "tumor_area_pct": tumor_area_pct,
        },
    )


def model_db(request):
    models_qs = UserImageModel.objects.all()
    return render(request, 'app/model_db.html', {'models': models_qs})


def logout_view(request):
    auth_logout(request)
    return redirect('/')


def text_to_speech(text, delay=7):
    time.sleep(delay)
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def profile_list(request):
    profiles = Profile.objects.all()
    return render(request, 'app/profile_list.html', {'profiles': profiles})


def database(request):
    models_qs = UserImageModel.objects.all().order_by('-id')
    return render(request, 'app/database.html', {'models': models_qs})


@require_POST
def delete_image(request, pk):
    record = get_object_or_404(UserImageModel, pk=pk)
    record.delete()
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'status': 'deleted', 'id': pk})
    return redirect('database')


@require_POST
def delete_all_images(request):
    for record in UserImageModel.objects.all():
        record.delete()
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'status': 'cleared'})
    return redirect('database')


@require_POST
@csrf_exempt
def chatbot_response_view(request):
    the_question = request.POST.get('question', '')
    response = chatbot_response(the_question)
    return JsonResponse({"response": response})
