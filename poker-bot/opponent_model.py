from collections import Counter


class OpponentModel:
    """
    Lightweight online opponent classifier.

    Tracks opponent action frequencies and maps them to a compact profile bucket:
    0 = unknown/balanced
    1 = call_station
    2 = aggressive
    3 = tight/passive
    4 = tricky/mixed

    Intended use:
    - call record_action(valid_actions, action) when opponent acts
    - pass profile_bucket() into QLearningAgent.get_state_key / choose_action / update
    """

    UNKNOWN = 0
    CALL_STATION = 1
    AGGRESSIVE = 2
    TIGHT_PASSIVE = 3
    TRICKY_MIXED = 4

    def __init__(self, min_actions=8):
        self.min_actions = min_actions
        self.action_counts = Counter()
        self.total_actions = 0
        self.facing_bet_count = 0
        self.call_when_facing_bet = 0
        self.fold_when_facing_bet = 0
        self.no_bet_count = 0
        self.bet_when_no_bet = 0
        self.check_when_no_bet = 0

    def reset(self):
        self.action_counts.clear()
        self.total_actions = 0
        self.facing_bet_count = 0
        self.call_when_facing_bet = 0
        self.fold_when_facing_bet = 0
        self.no_bet_count = 0
        self.bet_when_no_bet = 0
        self.check_when_no_bet = 0

    def record_action(self, valid_actions, action):
        self.action_counts[action] += 1
        self.total_actions += 1

        if 0 in valid_actions and 2 in valid_actions:
            self.facing_bet_count += 1
            if action == 0:
                self.call_when_facing_bet += 1
            elif action == 2:
                self.fold_when_facing_bet += 1

        if 0 in valid_actions and 1 in valid_actions:
            self.no_bet_count += 1
            if action == 1:
                self.bet_when_no_bet += 1
            elif action == 0:
                self.check_when_no_bet += 1

    def call_rate(self):
        if self.facing_bet_count == 0:
            return 0.0
        return self.call_when_facing_bet / self.facing_bet_count

    def fold_rate(self):
        if self.facing_bet_count == 0:
            return 0.0
        return self.fold_when_facing_bet / self.facing_bet_count

    def bet_rate(self):
        if self.no_bet_count == 0:
            return 0.0
        return self.bet_when_no_bet / self.no_bet_count

    def check_rate(self):
        if self.no_bet_count == 0:
            return 0.0
        return self.check_when_no_bet / self.no_bet_count

    def profile_bucket(self):
        if self.total_actions < self.min_actions:
            return self.UNKNOWN

        call_rate = self.call_rate()
        fold_rate = self.fold_rate()
        bet_rate = self.bet_rate()

        if call_rate >= 0.78 and fold_rate <= 0.22:
            return self.CALL_STATION

        if bet_rate >= 0.55:
            return self.AGGRESSIVE

        if fold_rate >= 0.50 and bet_rate <= 0.25:
            return self.TIGHT_PASSIVE

        if 0.35 <= call_rate <= 0.75 and 0.20 <= bet_rate <= 0.55:
            return self.TRICKY_MIXED

        return self.UNKNOWN

    def features(self):
        return {
            "profile_bucket": self.profile_bucket(),
            "total_actions": self.total_actions,
            "call_rate": self.call_rate(),
            "fold_rate": self.fold_rate(),
            "bet_rate": self.bet_rate(),
        }