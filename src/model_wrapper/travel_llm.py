import os

from src.traveluav_qwen.model_wrapper.travel_qwen import TravelQwenWrapper


class TravelModelWrapper(TravelQwenWrapper):

    def __init__(self, model_args, data_args):
        super().__init__(model_args=model_args, data_args=data_args)
        self.stop_mode = "stop_prob"
        self.stop_prob_threshold = float(os.environ.get("UAV_QWEN_STOP_PROB_THRESH", "0.5"))
        self.stop_prob_consecutive = max(1, int(os.environ.get("UAV_QWEN_STOP_PROB_CONSECUTIVE", "2")))
        self._stop_prob_streaks = []
        self.last_predict_done_details = []

    def _predict_done_by_stop_prob(self, episodes):
        bs = len(episodes)
        if len(self._stop_prob_streaks) != bs:
            self._stop_prob_streaks = [0 for _ in range(bs)]
        pred_stop_probs = getattr(self, "last_pred_stop_probs", None)
        if pred_stop_probs is None:
            return [False for _ in range(bs)], [{} for _ in range(bs)]

        stop_dones = []
        stop_details = []
        for i in range(bs):
            if len(episodes[i]) <= 1:
                self._stop_prob_streaks[i] = 0

            if i >= len(pred_stop_probs):
                self._stop_prob_streaks[i] = 0
                stop_dones.append(False)
                stop_details.append(
                    {
                        "pred_stop_prob": None,
                        "stop_prob_condition_met": False,
                        "stop_prob_streak": 0,
                        "stop_prob_triggered": False,
                    }
                )
                continue

            p = float(pred_stop_probs[i])
            cond_met = bool((p == p) and (abs(p) != float("inf")) and p >= self.stop_prob_threshold)
            if cond_met:
                self._stop_prob_streaks[i] += 1
            else:
                self._stop_prob_streaks[i] = 0
            triggered = bool(self._stop_prob_streaks[i] >= self.stop_prob_consecutive)
            stop_dones.append(triggered)
            stop_details.append(
                {
                    "pred_stop_prob": p,
                    "stop_prob_condition_met": cond_met,
                    "stop_prob_streak": int(self._stop_prob_streaks[i]),
                    "stop_prob_triggered": triggered,
                }
            )
        return stop_dones, stop_details

    def predict_done(self, episodes, object_infos):
        bs = len(episodes)
        final, stop_details = self._predict_done_by_stop_prob(episodes)

        details = []
        for i in range(bs):
            stop_detail = stop_details[i] if i < len(stop_details) else {}
            details.append(
                {
                    "stop_mode": self.stop_mode,
                    "predict_done": bool(final[i]) if i < len(final) else False,
                    "predict_done_by_stop": bool(final[i]) if i < len(final) else False,
                    "pred_stop_prob": stop_detail.get("pred_stop_prob"),
                    "stop_prob_condition_met": bool(stop_detail.get("stop_prob_condition_met", False)),
                    "stop_prob_streak": int(stop_detail.get("stop_prob_streak", 0)),
                    "stop_prob_triggered": bool(stop_detail.get("stop_prob_triggered", False)),
                    "stop_prob_threshold": float(self.stop_prob_threshold),
                    "stop_prob_consecutive": int(self.stop_prob_consecutive),
                }
            )
        self.last_predict_done_details = details
        return final
