from __future__ import annotations

from rest_framework import generics, permissions

from .models import SecureSetting
from .serializers import SecureSettingSerializer


class SecureSettingListCreateView(generics.ListCreateAPIView):
    serializer_class = SecureSettingSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return SecureSetting.objects.filter(user=self.request.user).order_by("category", "name")


class SecureSettingDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = SecureSettingSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return SecureSetting.objects.filter(user=self.request.user)
