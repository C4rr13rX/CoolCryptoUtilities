from django.contrib import admin

from .models import Payout, PayoutDestination


@admin.register(PayoutDestination)
class PayoutDestinationAdmin(admin.ModelAdmin):
    list_display = ("name", "kind", "chain", "asset_symbol", "share",
                    "min_net_profit_usd", "enabled", "auto_send")
    list_filter = ("kind", "chain", "enabled", "auto_send")
    search_fields = ("name", "address")


@admin.register(Payout)
class PayoutAdmin(admin.ModelAdmin):
    list_display = ("created_at", "destination", "amount_usd", "status",
                    "trades_counted", "tx_hash")
    list_filter = ("status", "destination")
    readonly_fields = ("profit_window_start_ts", "profit_window_end_ts")
