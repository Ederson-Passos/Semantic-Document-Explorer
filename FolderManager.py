from pathlib import Path


def check_directory_existence(temp_dir):
    """
    Checks if a directory exists and creates it if it doesn't.

    Args:
        temp_dir (Path): caminho para checar/criar o diretório.
    Raises:
        OSError: If there is an error creating the directory.
    """
    if not temp_dir.exists():  # Checa se o diretório existe.
        print(f"Diretório '{temp_dir}' não encontrado. Criando...")
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)  # Criado o diretório, incluindo pais se necessário.
            print(f"Diretório '{temp_dir}' criado com sucesso.")
        except OSError as e:
            raise OSError(f"Erro ao criar o diretório '{temp_dir}': {e}") from e
    elif not temp_dir.is_dir():  # Checa se não é um diretório.
        print(f"Erro: '{temp_dir}' existe, mas não é um diretório.")
    else:
        print(f"Diretório '{temp_dir}' já existe.")
