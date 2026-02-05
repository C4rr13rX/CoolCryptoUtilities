from __future__ import annotations

from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from addressbook.models import AddressBookEntry
from addressbook.serializers import AddressBookEntrySerializer


class AddressBookEntryListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = AddressBookEntrySerializer

    def get_queryset(self):
        return AddressBookEntry.objects.filter(user=self.request.user)

    def perform_create(self, serializer: AddressBookEntrySerializer) -> None:
        serializer.save(user=self.request.user)

    def get_serializer_context(self) -> dict:
        context = super().get_serializer_context()
        context["request"] = self.request
        return context


class AddressBookEntryDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = AddressBookEntrySerializer

    def get_queryset(self):
        return AddressBookEntry.objects.filter(user=self.request.user)

    def get_serializer_context(self) -> dict:
        context = super().get_serializer_context()
        context["request"] = self.request
        return context


class AddressBookLookupView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        name = (request.query_params.get("name") or request.query_params.get("q") or "").strip()
        exact = (request.query_params.get("exact") or "").strip().lower() in {"1", "true", "yes", "on"}
        limit = int(request.query_params.get("limit") or 10)
        if not name:
            return Response({"detail": "name is required"}, status=status.HTTP_400_BAD_REQUEST)
        queryset = AddressBookEntry.objects.filter(user=request.user)
        if exact:
            queryset = queryset.filter(name__iexact=name)
        else:
            queryset = queryset.filter(name__icontains=name)
        results = queryset.order_by("name", "address")[: max(1, min(limit, 50))]
        serializer = AddressBookEntrySerializer(results, many=True, context={"request": request})
        return Response({"count": queryset.count(), "results": serializer.data}, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        name = str((request.data or {}).get("name") or (request.data or {}).get("q") or "").strip()
        if not name:
            return Response({"detail": "name is required"}, status=status.HTTP_400_BAD_REQUEST)
        exact = bool((request.data or {}).get("exact", False))
        limit = int((request.data or {}).get("limit") or 10)
        queryset = AddressBookEntry.objects.filter(user=request.user)
        if exact:
            queryset = queryset.filter(name__iexact=name)
        else:
            queryset = queryset.filter(name__icontains=name)
        results = queryset.order_by("name", "address")[: max(1, min(limit, 50))]
        serializer = AddressBookEntrySerializer(results, many=True, context={"request": request})
        return Response({"count": queryset.count(), "results": serializer.data}, status=status.HTTP_200_OK)
