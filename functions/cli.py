import click
import subprocess
import sys
import os
import importlib.resources as pkg_resources

SCRIPT_MAP = {
    "attention_graph": "attention_graph.py",
    "structure_inf": "structure_inf.py",
    "position_inf": "position_inf.py",
    "instinct_inf": "instinct_inf.py",
    "model_train": "model_train.py",
    "model_predict": "model_predict.py",
    "seq_motifs": "seq_motifs.py",
    "structure_motifs": "structure_motifs.py",
    "high_attention_region": "high_attention_region.py",
}


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
    """NoMoCLIP: Interpretable Modeling of RNA–Protein Interactions from eCLIP‑Seq Profiles for Motif‑Free RBPs"""
    pass


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("task", type=click.Choice(SCRIPT_MAP.keys()))
@click.option("--env", help="Conda environment name (if not provided, run in current env)")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)  # 捕获额外参数传递给脚本
def run(task, env, args):
    """
    Run a task in the specified environment

    Available tasks:

        attention_graph       Semantic encoding.

        structure_inf         Structural encoding. This feature requires the RNAplfold tool, which is executed in a Python 2.7 environment. Please set the --env parameter to the local RNAplfold environment.

        position_inf          Sequential encoding.

        instinct_inf          Functional properties. For this feature, you need to use the corain. Please set the --env parameter to the local corain environment.

        model_train           Training Process.

        model_predict         Prediction.

        seq_motifs            Sequential motifs.

        structure_motifs      Structural motifs.

        high_attention_region High attention regions.

    Usage examples:

        nomoclip run position_inf --set_path test.fa --out_path outdir

        nomoclip run structure_inf --env RNAfold -- --set_path test.fa --out_path outdir
    """

    # Print description for task in help
    if '--help' in args or '-h' in args:
        script, desc = SCRIPT_MAP[task]
        click.echo(f"{task}: {desc}")
        sys.exit(0)

    script = SCRIPT_MAP[task]
    script_path = os.path.join(os.path.dirname(__file__), script)

    if not os.path.exists(script_path):
        click.echo(f"[ERROR] Script not found: {script_path}")
        sys.exit(1)

    if env is None:
        cmd = [sys.executable, script_path] + list(args)
    else:
        cmd = ["conda", "run", "-n", env, "python", script_path] + list(args)

    click.echo(f"[INFO] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"[ERROR] Failed to run {task} in env {env if env else 'current'}: {e}")
        sys.exit(1)


@click.command()
def install():
    """Run dependency installation script in sequence"""
    with pkg_resources.path("nomoclip", "install.sh") as script:
        subprocess.run(["bash", str(script)], check=True)

cli.add_command(run)
cli.add_command(install)

