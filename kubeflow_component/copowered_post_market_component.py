from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model


# -----------------------
# Step 1: Download Repo
# -----------------------
@dsl.component(base_image="python:3.13-slim")
def download_repo(
    github_repo_url: str,
    project_files: Output[Model],
    data: Output[Dataset],
    branch: str = "main",
) -> None:
    """Download specific scripts and data from a GitHub repository.
    This component clones a GitHub repository, copies selected Python scripts
    from the src/ folder into the `project_files` output, and the `data` folder
    into the `data` output.

    Args:
        github_repo_url (str): URL of the GitHub repository to clone.
        project_files (Output[Model]): Output path for project scripts.
        data (Output[Dataset]): Output path for data folder.
        branch (str): Branch name to pull from (defaults to 'main').
    """
    import shutil
    from pathlib import Path
    import subprocess

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


# -----------------------
# Step 2: Expert Knowledge Evaluation
# -----------------------
@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=["pandas==2.3.2", "numpy==2.3.3"],
)
def expert_knowledge_evaluation(
    project_files: Input[Model],
    data: Input[Dataset],
    expert_knowledge_results: Output[Dataset],
) -> None:
    """Runs expert knowledge evaluation on synthetic data against clinical ranges.

    Args:
        project_files (Input[Model]): Input containing project scripts from repository.
        data (Input[Dataset]): Input dataset containing synthetic data files.
        expert_knowledge_results (Output[Dataset]): Output path for expert knowledge results.
    """
    from pathlib import Path
    import subprocess

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
    base_image="python:3.13-slim",
    packages_to_install=["pandas==2.3.2", "numpy==2.3.3"],
)
def statistical_analysis(
    project_files: Input[Model],
    data: Input[Dataset],
    statistical_results: Output[Dataset],
) -> None:
    """Runs comprehensive statistical analysis on synthetic data for quality assessment.

    Args:
        project_files (Input[Model]): Input containing project scripts from repository.
        data (Input[Dataset]): Input dataset containing synthetic data files.
        statistical_results (Output[Dataset]): Output path for statistical analysis results.
    """
    from pathlib import Path
    import subprocess

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
@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=[
        "pandas==2.3.2",
        "numpy==2.3.3",
        "requests==2.32.5",
        "scikit-learn==1.7.2",
    ],
)
def adversarial_evaluation(
    project_files: Input[Model],
    data: Input[Dataset],
    adversarial_evaluation_results: Output[Dataset],
) -> None:
    """This component executes adversarial evaluation.
    Runs adversarial evaluation comparing synthetic vs real-world data performance.

    Args:
        project_files (Input[Model]): Input containing project scripts from repository.
        data (Input[Dataset]): Input dataset containing synthetic and real-world data files.
        adversarial_evaluation_results (Output[Dataset]): Output path for adversarial evaluation results.
    """
    from pathlib import Path
    import subprocess

    # Prepare paths
    proj_path = Path(project_files.path)
    data_path = Path(data.path)
    results_path = Path(adversarial_evaluation_results.path)
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare script and arguments
    script = proj_path / "adversarial_evaluation.py"
    if not script.exists():
        raise FileNotFoundError(f"Adversarial Evaluation script not found at {script}")

    cmd = [
        "python",
        str(script),
        "--synth_data",
        str(data_path / "synth_data.csv"),
        "--rwd_data",
        str(data_path / "rwd_data.csv"),
        "--output",
        str(results_path / "adversarial_evaluation_results.json"),
    ]
    subprocess.run(cmd, check=True)

    print(f"Adversarial Evaluation finished. Results saved to {results_path}")


# -----------------------
# -----------------------
# Define Pipeline
# -----------------------
# -----------------------
@dsl.pipeline(
    name="COPowereD Post-Market Evaluation Pipeline",
    description="Runs expert knowledge, statistical analysis, and adversarial evaluation checks.",
)
def copowered_post_market_pipeline(
    github_repo_url: str,
    branch: str = "main",
):
    """COPowereD Post-Market Evaluation Pipeline for synthetic data validation.
    This pipeline performs comprehensive post-market evaluation of COPowereD synthetic
    data through three parallel evaluation components: expert knowledge validation
    against clinical ranges, statistical quality analysis, and adversarial evaluation
    comparing synthetic vs real-world model performance.

    Args:
        github_repo_url (str): URL of the GitHub repository containing evaluation scripts.
        branch (str): Git branch to pull from repository (defaults to 'main').
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
    adversarial_task.set_cpu_request("1000m")
    adversarial_task.set_cpu_limit("2000m")
    adversarial_task.set_memory_request("2Gi")
    adversarial_task.set_memory_limit("4Gi")


if __name__ == "__main__":
    compiler = compiler.Compiler()
    compiler.compile(
        pipeline_func=copowered_post_market_pipeline,
        package_path="copowered_post_market_pipeline.yaml",
    )
