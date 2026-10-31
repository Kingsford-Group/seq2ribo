# sourced by every job script
# conda environment: S2R_ENV (default seq2ribo, see the top-level README)
set -eo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${S2R_ENV:-seq2ribo}"
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
mkdir -p logs
log() { echo "[$(date '+%F %T')] $(hostname) $*"; }
