(function () {
    const ATTEMPT_PREFIX = "sqg.quizAttempt:";

    function storageKey(quizPath) {
        return `${ATTEMPT_PREFIX}${String(quizPath || "")}`;
    }

    function emptyAnswers(questionCount) {
        return Array.from({ length: Math.max(0, Number(questionCount) || 0) }, () => null);
    }

    function buildAttempt(questionCount) {
        return {
            answers: emptyAnswers(questionCount),
            currentIndex: 0,
            completed: false,
            completedAt: "",
            score: null,
            total: Number(questionCount) || 0,
        };
    }

    function normalizeAttempt(rawAttempt, questionCount) {
        const total = Math.max(0, Number(questionCount) || 0);
        const base = buildAttempt(total);
        if (!rawAttempt || typeof rawAttempt !== "object") {
            return base;
        }

        const rawAnswers = Array.isArray(rawAttempt.answers) ? rawAttempt.answers : [];
        base.answers = emptyAnswers(total).map((_, index) => {
            const value = typeof rawAnswers[index] === "string" ? rawAnswers[index].toUpperCase() : null;
            return ["A", "B", "C", "D"].includes(value) ? value : null;
        });

        const rawIndex = Number.isInteger(rawAttempt.currentIndex) ? rawAttempt.currentIndex : 0;
        base.currentIndex = Math.min(Math.max(rawIndex, 0), Math.max(total - 1, 0));
        base.completed = rawAttempt.completed === true;
        base.completedAt = typeof rawAttempt.completedAt === "string" ? rawAttempt.completedAt : "";
        base.score = typeof rawAttempt.score === "number" ? rawAttempt.score : null;
        base.total = total;
        return base;
    }

    function loadAttempt(quizPath, questionCount) {
        try {
            const rawValue = window.localStorage.getItem(storageKey(quizPath));
            if (!rawValue) {
                return buildAttempt(questionCount);
            }
            return normalizeAttempt(JSON.parse(rawValue), questionCount);
        } catch (error) {
            console.error("Failed to load quiz attempt", error);
            return buildAttempt(questionCount);
        }
    }

    function saveAttempt(quizPath, attempt, questionCount) {
        const normalized = normalizeAttempt(attempt, questionCount);
        window.localStorage.setItem(storageKey(quizPath), JSON.stringify(normalized));
        return normalized;
    }

    function clearAttempt(quizPath) {
        window.localStorage.removeItem(storageKey(quizPath));
    }

    window.SQGQuizStore = {
        buildAttempt,
        clearAttempt,
        loadAttempt,
        normalizeAttempt,
        saveAttempt,
    };
}());
