#!/usr/bin/env bash
# dev_up.sh — one-command local dev launch: ChromaDB (Docker) + API + UI.
#
# Ce script lance tout l'environnement de développement local en une seule commande.
# Il vérifie si ChromaDB tourne (et le lance via Docker Compose sinon),
# puis démarre l'API FastAPI et l'interface Streamlit en arrière-plan.
# Un simple Ctrl+C permet d'arrêter l'API et l'UI proprement.

# Arrête l'exécution du script en cas d'erreur (-e), de variable non définie (-u) ou d'erreur dans un pipe (-o pipefail)
set -euo pipefail

# Se positionne à la racine du projet (le dossier parent de 'scripts')
cd "$(dirname "$0")/.."

# Vérifie que le script s'exécute sur un Mac avec puce Apple Silicon (arm64), indispensable pour MLX
if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: this stack requires Apple Silicon (MLX)." >&2
    exit 1
fi

# Vérifie que l'environnement virtuel Python est bien installé (créé via 'make setup')
if [[ ! -x .venv/bin/python ]]; then
    echo "ERROR: no venv found. Run 'make setup' first." >&2
    exit 1
fi

# Charge les variables d'environnement depuis le fichier .env s'il existe
# shellcheck disable=SC1091
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# Définit les ports par défaut si non spécifiés dans l'environnement
CHROMA_PORT="${CHROMA_PORT:-8001}"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

# Force l'utilisation des modèles locaux (déjà téléchargés) pour éviter que
# huggingface_hub ne tente de se connecter à internet à chaque démarrage.
# L'application reste ainsi 100% hors-ligne.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Sans ça, les logs de l'API/UI restent bufferisés dans le fichier tant que
# le process tourne — impossible de voir ce qui se passe pendant le chargement.
export PYTHONUNBUFFERED=1

# PyArrow (utilisé par Streamlit pour sérialiser les DataFrames vers le
# frontend, et par `datasets`/ragas côté API) embarque son propre
# allocateur mémoire (mimalloc). Ce build crashe (SIGSEGV, null deref
# dans mi_thread_init) quand il est touché pour la 1ère fois depuis un
# thread non-principal — exactement le thread interne de Streamlit
# (ScriptRunner), qui n'est jamais le main thread du process. Forcer
# l'allocateur système contourne complètement ce chemin de code buggé.
# Reproduit et vérifié : sans ça, l'onglet Index (bar chart "Distribution
# par source") segfault le process Streamlit de façon systématique.
export ARROW_DEFAULT_MEMORY_POOL=system

# Combien de temps on attend qu'API/UI répondent avant de considérer que
# le démarrage a échoué (cold start MLX + bge-m3 peut être lent la 1ère fois).
READY_TIMEOUT="${READY_TIMEOUT:-180}"

# Crée le dossier pour stocker les fichiers de logs de l'API et de l'UI
mkdir -p logs
: > logs/dev_api.log
: > logs/dev_ui.log

echo ">> Checking ChromaDB on :${CHROMA_PORT}..."
# Tente de joindre le endpoint heartbeat de ChromaDB pour voir s'il est actif
if ! curl -fsS "http://localhost:${CHROMA_PORT}/api/v2/heartbeat" >/dev/null 2>&1; then
    echo ">> Not running — starting via docker compose..."
    make chroma-up
    # Attend jusqu'à 30 secondes que ChromaDB soit prêt à répondre
    for i in $(seq 1 30); do
        curl -fsS "http://localhost:${CHROMA_PORT}/api/v2/heartbeat" >/dev/null 2>&1 && break
        sleep 1
        if [[ "$i" -eq 30 ]]; then
            echo "ERROR: ChromaDB did not come up in 30s. Check 'make chroma-logs'." >&2
            exit 1
        fi
    done
fi
echo ">> ChromaDB OK."

# Lance l'API FastAPI en arrière-plan et redirige ses sorties vers un fichier de log
echo ">> Starting API on :${API_PORT} (log: logs/dev_api.log)..."
.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port "${API_PORT}" > logs/dev_api.log 2>&1 &
API_PID=$! # Sauvegarde l'identifiant du processus (PID) de l'API

# Lance l'interface Streamlit en arrière-plan et redirige ses sorties
echo ">> Starting UI on :${UI_PORT} (log: logs/dev_ui.log)..."
PYTHONPATH=. .venv/bin/streamlit run src/ui/streamlit_app.py \
    --server.port "${UI_PORT}" --server.address 0.0.0.0 \
    > logs/dev_ui.log 2>&1 &
UI_PID=$! # Sauvegarde le PID de l'UI

# Fonction appelée automatiquement à l'arrêt du script (ex: via Ctrl+C)
# Elle tue les processus de l'API et de l'UI proprement en quittant
cleanup() {
    echo
    echo ">> Stopping API (${API_PID}) and UI (${UI_PID})..."
    kill "${API_PID}" "${UI_PID}" 2>/dev/null || true
    wait "${API_PID}" "${UI_PID}" 2>/dev/null || true
    echo ">> Stopped. ChromaDB left running (make chroma-down to stop it too)."
}
# Enregistre la fonction 'cleanup' pour qu'elle s'exécute à la sortie du script ou lors d'un signal d'interruption
trap cleanup EXIT INT TERM

# Attend qu'une URL réponde, avec un vrai résultat (0=up, 1=timeout) — pas
# de silence, pas de faux "c'est prêt" si ça n'a jamais répondu.
wait_for() {
    local name="$1" url="$2" pid="$3" log="$4"
    local waited=0
    echo -n ">> Waiting for ${name} on ${url} "
    while ! curl -fsS -m 2 "$url" >/dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo
            echo "XX ${name} process died before answering. Last log lines:"
            tail -30 "$log" 2>&1
            return 1
        fi
        if [[ "$waited" -ge "$READY_TIMEOUT" ]]; then
            echo
            echo "XX ${name} did not answer within ${READY_TIMEOUT}s. Last log lines:"
            tail -30 "$log" 2>&1
            echo "   (still running as PID ${pid} — check logs/${log##*/} for progress)"
            return 1
        fi
        echo -n "."
        sleep 3
        waited=$((waited + 3))
    done
    echo " OK (${waited}s)"
    return 0
}

echo
API_OK=0
UI_OK=0
wait_for "API" "http://localhost:${API_PORT}/health" "${API_PID}" "logs/dev_api.log" && API_OK=1
wait_for "UI"  "http://localhost:${UI_PORT}" "${UI_PID}" "logs/dev_ui.log" && UI_OK=1

echo
echo "=================================================================="
if [[ "$API_OK" -eq 1 ]]; then
    echo " API : http://localhost:${API_PORT}/docs  [UP]"
else
    echo " API : http://localhost:${API_PORT}/docs  [NOT RESPONDING — see above]"
fi
if [[ "$UI_OK" -eq 1 ]]; then
    echo " UI  : http://localhost:${UI_PORT}  [UP]"
else
    echo " UI  : http://localhost:${UI_PORT}  [NOT RESPONDING — see above]"
fi
echo "=================================================================="
echo " Ctrl+C to stop API + UI (ChromaDB stays up)."
echo

if [[ "$API_OK" -eq 0 && "$UI_OK" -eq 0 ]]; then
    echo "XX Neither service came up — stopping." >&2
    exit 1
fi

# Garde le script principal actif en attendant la fin des processus enfants
wait "${API_PID}" "${UI_PID}"
