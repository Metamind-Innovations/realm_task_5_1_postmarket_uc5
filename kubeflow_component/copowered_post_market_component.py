from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Model, Output

PYTHON_BASE_IMAGE = "python:3.14-slim"
# Insert the Docker Hub model image below.
# Example: "docker.io/<username>/<image_name>:<tag>"
COPOWERED_MODEL_IMAGE = "<docker_image>"


@dsl.component(base_image=PYTHON_BASE_IMAGE)
def download_repo(
        github_repo_url: str,
        project_files: Output[Model],
        data: Output[Dataset],
        branch: str = "main",
) -> None:
    """Download project scripts and data from a GitHub repository.

    This component clones a GitHub repository, copies selected Python scripts
    from the ``src`` folder into ``project_files``, and copies the ``data``
    folder into ``data``.

    :param github_repo_url: URL of the GitHub repository to clone.
    :param project_files: Output path for project scripts.
    :param data: Output path for input data files.
    :param branch: Branch name to clone.
    :raises FileNotFoundError: If a required data file is missing.
    """
    import shutil
    import subprocess
    from pathlib import Path

    repo_dir = Path("/tmp/repo")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    print("Installing git...")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "git"], check=True)

    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            branch,
            "--single-branch",
            github_repo_url,
            str(repo_dir),
        ],
        check=True,
    )
    print(f"Cloned repo {github_repo_url} (branch: {branch}).")

    # List of files to copy from src/ folder
    files_to_copy = [
        "adversarial_evaluation.py",
        "COPowereD_model.py",
        "config.json",
        "expert_knowledge.py",
        "statistical_analysis.py",
        "utils.py",
    ]

    # Copy specific scripts from src/ folder
    proj_path = Path(project_files.path)
    proj_path.mkdir(parents=True, exist_ok=True)
    src_folder = repo_dir / "src"

    for filename in files_to_copy:
        src_file = src_folder / filename
        if src_file.exists():
            shutil.copy2(src_file, proj_path / filename)
            print(f"Copied {filename} to project_files")
        else:
            print(f"Warning: {filename} not found in src/ folder")

    # Copy data folder
    data_path = Path(data.path)
    data_path.mkdir(parents=True, exist_ok=True)
    src_data_path = repo_dir / "data"
    if src_data_path.exists():
        shutil.copytree(src_data_path, data_path, dirs_exist_ok=True)
        print("Copied data folder")
    else:
        print("Warning: data folder not found in repo")

    for required_file in ("synth_data.csv", "rwd_data.csv"):
        if not (data_path / required_file).exists():
            raise FileNotFoundError(f"Missing required data file: {required_file}")


# -----------------------
# Step 2: Expert Knowledge Evaluation
# -----------------------
@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=["pandas>=2.3.3", "numpy>=2.3.3"],
)
def expert_knowledge_evaluation(
        project_files: Input[Model],
        data: Input[Dataset],
        expert_knowledge_results: Output[Dataset],
) -> None:
    """Run expert knowledge evaluation on synthetic data.

    :param project_files: Input containing project scripts from the repository.
    :param data: Input dataset containing synthetic data files.
    :param expert_knowledge_results: Output path for expert knowledge results.
    :raises FileNotFoundError: If the evaluation script is missing.
    """
    import subprocess
    from pathlib import Path

    # Prepare paths
    proj_path = Path(project_files.path)
    data_path = Path(data.path)
    results_path = Path(expert_knowledge_results.path)
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare script and arguments
    script = proj_path / "expert_knowledge.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Expert Knowledge evaluation script not found at {script}"
        )

    cmd = [
        "python",
        str(script),
        "--synth_data",
        str(data_path / "synth_data.csv"),
        "--output",
        str(results_path / "expert_knowledge_evaluation_results.json"),
    ]
    subprocess.run(cmd, check=True)

    print(f"Expert Knowledge evaluation finished. Results saved to {results_path}")


# -----------------------
# Step 3: Statistical Analysis
# -----------------------
@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=["pandas>=2.3.3", "numpy>=2.3.3"],
)
def statistical_analysis(
        project_files: Input[Model],
        data: Input[Dataset],
        statistical_results: Output[Dataset],
) -> None:
    """Run statistical analysis on synthetic data.

    :param project_files: Input containing project scripts from the repository.
    :param data: Input dataset containing synthetic data files.
    :param statistical_results: Output path for statistical analysis results.
    :raises FileNotFoundError: If the statistical analysis script is missing.
    """
    import subprocess
    from pathlib import Path

    # Prepare paths
    proj_path = Path(project_files.path)
    data_path = Path(data.path)
    results_path = Path(statistical_results.path)
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare script and arguments
    script = proj_path / "statistical_analysis.py"
    if not script.exists():
        raise FileNotFoundError(f"Statistical analysis script not found at {script}")

    cmd = [
        "python",
        str(script),
        "--synth_data",
        str(data_path / "synth_data.csv"),
        "--output",
        str(results_path / "statistical_analysis_results.json"),
    ]
    subprocess.run(cmd, check=True)

    print(f"Statistical analysis finished. Results saved to {results_path}")


# -----------------------
# Step 4: Adversarial Evaluation
# -----------------------
@dsl.container_component
def adversarial_evaluation(
        project_files: Input[Model],
        data: Input[Dataset],
        adversarial_evaluation_results: Output[Dataset],
) -> dsl.ContainerSpec:
    """Run adversarial evaluation inside the Dockerized COPowereD model image.

    The downloaded ``adversarial_evaluation.py`` script calls the local
    COPowereD model jar available in the container image.

    :param project_files: Input containing project scripts from the repository.
    :param data: Input dataset containing synthetic and real-world data files.
    :param adversarial_evaluation_results: Output path for adversarial results.
    :return: Container specification for the adversarial evaluation step.
    """

    command_str = f"""
        set -e
        export DEBIAN_FRONTEND=noninteractive
        output_dir="{adversarial_evaluation_results.path}"
        output_file="$output_dir/adversarial_evaluation_results.json"
        apt-get update
        apt-get install -y --no-install-recommends python3 python3-pip
        python3 -m pip install --no-cache-dir \
            'pandas==2.3.3' \
            'numpy>=1.26.4,<2.3' \
            'scikit-learn==1.7.2'
        mkdir -p "$output_dir"
        test -f "{project_files.path}/adversarial_evaluation.py"
        COPOWERED_MODEL_EXECUTION_MODE=local \
            python3 "{project_files.path}/adversarial_evaluation.py" \
                --synth_data "{data.path}/synth_data.csv" \
                --rwd_data "{data.path}/rwd_data.csv" \
                --output "$output_file"
    """

    return dsl.ContainerSpec(
        image=COPOWERED_MODEL_IMAGE,
        command=["sh", "-c"],
        args=[command_str],
    )


# -----------------------
# Define Pipeline
# -----------------------
@dsl.pipeline(
    name="COPowereD Post-Market Evaluation Pipeline",
    description=(
            "Runs expert knowledge, statistical analysis, and adversarial "
            "evaluation checks."
    ),
)
def copowered_post_market_pipeline(
        github_repo_url: str,
        branch: str = "main",
) -> None:
    """Run the COPowereD post-market evaluation pipeline.

    :param github_repo_url: URL of the GitHub repository containing evaluation
        scripts and data.
    :param branch: Git branch to clone from the repository.
    """
    repo_task = download_repo(github_repo_url=github_repo_url, branch=branch)
    repo_task.set_caching_options(False)
    repo_task.set_cpu_request("1000m")
    repo_task.set_cpu_limit("2000m")
    repo_task.set_memory_request("2Gi")
    repo_task.set_memory_limit("4Gi")

    expert_knowledge_task = expert_knowledge_evaluation(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
    )
    expert_knowledge_task.after(repo_task)
    expert_knowledge_task.set_caching_options(False)
    expert_knowledge_task.set_cpu_request("1000m")
    expert_knowledge_task.set_cpu_limit("2000m")
    expert_knowledge_task.set_memory_request("2Gi")
    expert_knowledge_task.set_memory_limit("4Gi")

    statistical_task = statistical_analysis(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
    )
    statistical_task.after(repo_task)
    statistical_task.set_caching_options(False)
    statistical_task.set_cpu_request("1000m")
    statistical_task.set_cpu_limit("2000m")
    statistical_task.set_memory_request("2Gi")
    statistical_task.set_memory_limit("4Gi")

    adversarial_task = adversarial_evaluation(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
    )
    adversarial_task.after(repo_task)
    adversarial_task.set_caching_options(False)
    adversarial_task.set_cpu_request("4000m")
    adversarial_task.set_cpu_limit("8000m")
    adversarial_task.set_memory_request("6Gi")
    adversarial_task.set_memory_limit("10Gi")


if __name__ == "__main__":
    kfp_compiler = compiler.Compiler()
    kfp_compiler.compile(
        pipeline_func=copowered_post_market_pipeline,
        package_path="copowered_post_market_pipeline.yaml",
    )
