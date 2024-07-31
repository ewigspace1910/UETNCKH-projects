class Helper:
    @staticmethod
    def normalized_history(history):
        normalizedHistory = []
        for item in history:
            normalizedHistory.append((item.question,item.answer))

        return normalizedHistory
