"""
Sube la carpeta streamlit_cloud/ a un repositorio GitHub usando la API de GitHub.

Requisitos:
  pip install PyGithub

Uso:
  export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
  python Desarrollo/tfm-ecg/scripts/upload_to_streamlit_cloud.py \
      --repo TU_USUARIO/ecg-streamlit-cloud \
      --branch main

El script crea el repositorio si no existe y sube todos los ficheros de
streamlit_cloud/ manteniendo la estructura de directorios.

Después, ve a https://share.streamlit.io → "New app" → selecciona el repo
y pon app.py como fichero principal.
"""

import argparse
import base64
import os
from pathlib import Path

try:
    from github import Github, GithubException
except ImportError:
    raise SystemExit(
        "Instala PyGithub:  pip install PyGithub"
    )

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
STREAMLIT_DIR = Path(__file__).parent.parent / "streamlit_cloud"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_or_create_repo(g: Github, full_name: str):
    user_or_org, repo_name = full_name.split("/", 1)
    try:
        return g.get_repo(full_name)
    except GithubException:
        print(f"[GitHub] Creando repositorio {full_name} ...")
        user = g.get_user()
        repo = user.create_repo(
            name=repo_name,
            description="ECG Diagnosis AI — Streamlit Community Cloud demo",
            private=False,
            auto_init=True,
        )
        print(f"[GitHub] Repositorio creado: {repo.html_url}")
        return repo


def upload_file(repo, local_path: Path, remote_path: str, branch: str):
    content = local_path.read_bytes()
    try:
        existing = repo.get_contents(remote_path, ref=branch)
        repo.update_file(
            path=remote_path,
            message=f"update {remote_path}",
            content=content,
            sha=existing.sha,
            branch=branch,
        )
        print(f"  ↑ (update) {remote_path}")
    except GithubException:
        repo.create_file(
            path=remote_path,
            message=f"add {remote_path}",
            content=content,
            branch=branch,
        )
        print(f"  ↑ (create) {remote_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sube streamlit_cloud/ a GitHub")
    parser.add_argument(
        "--repo", required=True,
        help="Nombre completo del repositorio GitHub (usuario/nombre)",
    )
    parser.add_argument("--branch", default="main", help="Rama destino (default: main)")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit(
            "Falta la variable de entorno GITHUB_TOKEN.\n"
            "Crea un token en https://github.com/settings/tokens (scope: repo)"
        )

    g    = Github(token)
    repo = get_or_create_repo(g, args.repo)

    print(f"\n[GitHub] Subiendo ficheros a {args.repo} (branch: {args.branch})\n")
    for local_file in sorted(STREAMLIT_DIR.rglob("*")):
        if local_file.is_file() and "__pycache__" not in str(local_file):
            remote_path = local_file.relative_to(STREAMLIT_DIR).as_posix()
            upload_file(repo, local_file, remote_path, args.branch)

    print(f"\n✅ Subida completada.")
    print(f"   Repositorio: {repo.html_url}")
    print(f"\n👉 Siguiente paso:")
    print(f"   1. Ve a https://share.streamlit.io")
    print(f"   2. Pulsa 'New app'")
    print(f"   3. Selecciona el repo '{args.repo}', branch '{args.branch}', fichero 'app.py'")
    print(f"   4. (Opcional) Añade HF_TOKEN en 'Advanced settings → Secrets'")


if __name__ == "__main__":
    main()
