from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import UserImageModel


from .models import Profile

class RegisterForm(UserCreationForm):
    # fields we want to include and customize in our form
    first_name = forms.CharField(max_length=100,
                                 required=True,
                                 widget=forms.TextInput(attrs={'placeholder': 'First Name',
                                                               'class': 'form-control',
                                                               }))
    last_name = forms.CharField(max_length=100,
                                required=True,
                                widget=forms.TextInput(attrs={'placeholder': 'Last Name',
                                                              'class': 'form-control',
                                                              }))
    username = forms.CharField(max_length=100,
                               required=True,
                               widget=forms.TextInput(attrs={'placeholder': 'Username',
                                                             'class': 'form-control',
                                                             }))
    email = forms.EmailField(required=True,
                             widget=forms.TextInput(attrs={'placeholder': 'Email',
                                                           'class': 'form-control',
                                                           }))
    password1 = forms.CharField(max_length=50,
                                required=True,
                                widget=forms.PasswordInput(attrs={'placeholder': 'Password',
                                                                  'class': 'form-control',
                                                                  'data-toggle': 'password',
                                                                  'id': 'password',
                                                                  }))
    password2 = forms.CharField(max_length=50,
                                required=True,
                                widget=forms.PasswordInput(attrs={'placeholder': 'Confirm Password',
                                                                  'class': 'form-control',
                                                                  'data-toggle': 'password',
                                                                  'id': 'password',
                                                                  }))

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'username', 'email', 'password1', 'password2']


class LoginForm(AuthenticationForm):
    username = forms.CharField(max_length=100,
                               required=True,
                               widget=forms.TextInput(attrs={'placeholder': 'Username',
                                                             'class': 'form-control',
                                                             }))
    password = forms.CharField(max_length=50,
                               required=True,
                               widget=forms.PasswordInput(attrs={'placeholder': 'Password',
                                                                 'class': 'form-control',
                                                                 'data-toggle': 'password',
                                                                 'id': 'password',
                                                                 'name': 'password',
                                                                 }))
    remember_me = forms.BooleanField(required=False)

    class Meta:
        model = User
        fields = ['username', 'password', 'remember_me']


class UpdateUserForm(forms.ModelForm):
    username = forms.CharField(max_length=100,
                               required=True,
                               widget=forms.TextInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(required=True,
                             widget=forms.TextInput(attrs={'class': 'form-control'}))

    class Meta:
        model = User
        fields = ['username', 'email']


import base64
from io import BytesIO
from PIL import Image
from .models import DEFAULT_AVATAR_BASE64, Profile, UserImageModel, get_default_avatar


class UpdateProfileForm(forms.ModelForm):
    avatar = forms.FileField(
        required=False,
        widget=forms.FileInput(attrs={'class': 'form-control-file', 'id': 'id_avatar', 'accept': 'image/*'})
    )
    bio = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 5})
    )

    class Meta:
        model = Profile
        fields = ['avatar', 'bio']

    def clean_avatar(self):
        uploaded = self.cleaned_data.get('avatar')
        if uploaded and hasattr(uploaded, 'read'):
            try:
                img = Image.open(uploaded)
                img.thumbnail((256, 256))
                buf = BytesIO()
                fmt = "PNG" if img.mode == "RGBA" else "JPEG"
                mime = "image/png" if fmt == "PNG" else "image/jpeg"
                img.save(buf, format=fmt, quality=90)
                b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                return f"data:{mime};base64,{b64_str}"
            except Exception:
                pass
        # Retain existing avatar if no new file uploaded
        if self.instance and self.instance.avatar:
            return self.instance.avatar
        return get_default_avatar()


class UserImageForm(forms.ModelForm):
    image = forms.FileField(
        required=True,
        widget=forms.FileInput(attrs={'id': 'fileInput', 'accept': 'image/*'})
    )

    class Meta:
        model = UserImageModel
        fields = ['image']

    def clean_image(self):
        uploaded = self.cleaned_data.get('image')
        if uploaded and hasattr(uploaded, 'read'):
            file_bytes = uploaded.read()
            mime_type = getattr(uploaded, 'content_type', None) or 'image/jpeg'
            b64_str = base64.b64encode(file_bytes).decode('utf-8')
            return f"data:{mime_type};base64,{b64_str}"
        elif isinstance(uploaded, str):
            return uploaded
        return uploaded



