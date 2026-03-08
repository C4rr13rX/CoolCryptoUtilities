class BudgetTracker:
    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd
        self.spent_usd = 0.0
        self.enabled = True

    def add_spend(self, amount: float) -> None:
        if not self.enabled:
            return
        self.spent_usd += max(0.0, amount)

    def exceeded(self) -> bool:
        return self.enabled and self.spent_usd >= self.budget_usd

    def reset(self, new_budget: float | None = None) -> None:
        if new_budget is not None:
            self.budget_usd = new_budget
        self.spent_usd = 0.0

    def disable(self) -> None:
        self.enabled = False

