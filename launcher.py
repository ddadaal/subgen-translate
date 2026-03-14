import os
import sys
import urllib.request
import subprocess
import argparse


def infer_source_language_from_srt_path(srt_path):
    """Try to infer source language from subtitle filename tokens.

    Example: movie.subgen.medium.jpn.srt -> jpn
    """
    base_name = os.path.splitext(os.path.basename(srt_path))[0]
    tokens = base_name.split('.')

    try:
        from language_code import LanguageCode
    except Exception:
        return None

    for token in reversed(tokens):
        lang = LanguageCode.from_string(token)
        if lang:
            return lang.to_iso_639_1() or token

    return None

def convert_to_bool(in_bool):
    # Convert the input to string and lower case, then check against true values
    return str(in_bool).lower() in ('true', 'on', '1', 'y', 'yes')

def install_packages_from_requirements(requirements_file):
    try:
        subprocess.run(['pip3', 'install', '-r', requirements_file, '--upgrade'], check=True)
        print("Packages installed successfully using pip3.")
    except subprocess.CalledProcessError:
        try:
            subprocess.run(['pip', 'install', '-r', requirements_file, '--upgrade'], check=True)
            print("Packages installed successfully using pip.")
        except subprocess.CalledProcessError:
            print("Failed to install packages using both pip3 and pip.")

def download_from_github(url, output_file):
    try:
        with urllib.request.urlopen(url) as response, open(output_file, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"File downloaded successfully to {output_file}")
    except urllib.error.HTTPError as e:
        print(f"Failed to download file from {url}. HTTP Error Code: {e.code}")
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
    except Exception as e:
        print(f"An error occurred: {e}")

def prompt_and_save_bazarr_env_variables():
    instructions = (
        "You will be prompted for several configuration values.\n"
        "If you wish to use the default value for any of them, simply press Enter without typing anything.\n"
        "The default values are shown in brackets [] next to the prompts.\n"
        "Items can be the value of true, on, 1, y, yes, false, off, 0, n, no, or an appropriate text response.\n"
    )
    print(instructions)
    env_vars = {
        'WHISPER_MODEL': ('Whisper Model', 'Enter the Whisper model you want to run: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large, distil-large-v2, distil-medium.en, distil-small.en', 'medium'),
        'WEBHOOKPORT': ('Webhook Port', 'Default listening port for subgen.py', '9000'),
        'TRANSCRIBE_DEVICE': ('Transcribe Device', 'Set as cpu or gpu', 'gpu'),
        # Defaulting to False here for the prompt, user can change
        'DEBUG': ('Debug', 'Enable debug logging (true/false)', 'False'),
        'CLEAR_VRAM_ON_COMPLETE': ('Clear VRAM', 'Attempt to clear VRAM when complete (Windows users may need to set this to False)', 'False'),
        'APPEND': ('Append', 'Append \'Transcribed by whisper\' to generated subtitle (true/false)', 'False'),
    }

    user_input = {}
    with open('subgen.env', 'w') as file:
        for var, (description, prompt, default) in env_vars.items():
            value = input(f"{prompt} [{default}]: ") or default
            file.write(f"{var}={value}\n")
    print("Environment variables have been saved to subgen.env")

def load_env_variables(env_filename='subgen.env'):
    try:
        with open(env_filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    var, value = line.split('=', 1)
                    # Only set if not already set by a higher priority mechanism (like external env var)
                    # For this simple loader, we'll let it overwrite,
                    # and CLI args will overwrite these later if specified.
                    os.environ[var] = value
        print(f"Environment variables have been loaded from {env_filename}")
    except FileNotFoundError:
        print(f"{env_filename} file not found. Consider running with --setup-bazarr or creating it manually.")

def main():
    if 'python3' in sys.executable:
        python_cmd = 'python3'
    elif 'python' in sys.executable:
        python_cmd = 'python'
    else:
        print("Script started with an unknown command")
        sys.exit(1)
    if sys.version_info[0] < 3:
        print(f"This script requires Python 3 or higher, you are running {sys.version}")
        sys.exit(1)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(prog="python launcher.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Changed: action='store_true' means it's False by default, True if flag is present
    parser.add_argument('-d', '--debug', action='store_true', help="Enable console debugging (overrides .env and external ENV)")
    parser.add_argument('-i', '--install', action='store_true', help="Install/update all necessary packages")
    # Changed: action='store_true'
    parser.add_argument('-a', '--append', action='store_true', help="Append 'Transcribed by whisper' (overrides .env and external ENV)")
    parser.add_argument('-u', '--update', action='store_true', help="Update Subgen")
    parser.add_argument('-x', '--exit-early', action='store_true', help="Exit without running subgen.py")
    parser.add_argument('-s', '--setup-bazarr', action='store_true', help="Prompt for common Bazarr setup parameters and save them for future runs")
    parser.add_argument('-b', '--branch', type=str, default='main', help='Specify the branch to download from')
    parser.add_argument('-l', '--launcher-update', action='store_true', help="Update launcher.py and re-launch")
    parser.add_argument('-f', '--file', type=str, default=None, metavar='VIDEO_PATH', help='Path to a video/audio file to run the full transcription + translation pipeline directly (skips server launch)')
    parser.add_argument('-t', '--type', type=str, default=None, choices=['transcribe', 'translate'], metavar='TYPE', help='Transcription type for --file mode: "transcribe" or "translate" (default: from TRANSCRIBE_OR_TRANSLATE env var)')
    parser.add_argument('--language', type=str, default=None, metavar='LANGUAGE', help='Force transcription language (e.g. en, ja, zh). Overrides FORCE_DETECTED_LANGUAGE_TO env var.')
    parser.add_argument('-S', '--srt', type=str, default=None, metavar='SRT_PATH', help='Path to an existing .srt file to translate directly. Takes priority over --file if both are given.')
    parser.add_argument('--srt-to', type=str, default=None, metavar='LANGUAGE', help='Target language for --srt mode (e.g. zh, ja, Chinese). Falls back to TRANSLATE_TO env var.')
    parser.add_argument('--srt-source-language', type=str, default=None, metavar='LANGUAGE', help='Required source language for --srt mode (e.g. en, English).')

    args = parser.parse_args()

    branch_name = args.branch if args.branch != 'main' else os.getenv('BRANCH', 'main')
    script_name_suffix = f"-{branch_name}.py" if branch_name != "main" else ".py"
    subgen_script_to_run = f"subgen{script_name_suffix}"
    language_code_script_to_download = f"language_code{script_name_suffix}"


    if args.launcher_update or convert_to_bool(os.getenv('LAUNCHER_UPDATE')):
        print(f"Updating launcher.py from GitHub branch {branch_name}...")
        download_from_github(f"https://raw.githubusercontent.com/McCloudS/subgen/{branch_name}/launcher.py", f'launcher{script_name_suffix}')
        excluded_args = ['--launcher-update', '-l']
        new_args = [arg for arg in sys.argv[1:] if arg not in excluded_args]
        print(f"Relaunching updated launcher: launcher{script_name_suffix}")
        os.execl(sys.executable, sys.executable, f"launcher{script_name_suffix}", *new_args)
        # The script will not continue past os.execl

    # --- Environment Variable Handling ---
    # 1. Load from .env file first. This sets a baseline.
    #    External environment variables (set before launcher.py) will already be in os.environ
    #    and won't be overwritten by load_env_variables IF load_env_variables checked for existence.
    #    For simplicity, this version of load_env_variables *will* overwrite.
    #    If you need to preserve external env vars over .env, load_env_variables needs adjustment.
    if args.setup_bazarr:
        prompt_and_save_bazarr_env_variables()
        # After saving, load them immediately for this run
        load_env_variables()
        load_env_variables('subgen.env.local')
    else:
        # Load if not setting up, assuming subgen.env might exist
        load_env_variables()
        load_env_variables('subgen.env.local')


    # 2. Override with command-line arguments (highest priority for these specific flags)
    if args.debug: # If -d or --debug was passed
        os.environ['DEBUG'] = 'True'
        print("Launcher CLI: DEBUG set to True")
    elif 'DEBUG' not in os.environ: # If not set by CLI and not by .env or external
        os.environ['DEBUG'] = 'False' # Default to False if nothing else specified it
        print("Launcher: DEBUG defaulted to False (no prior setting)")


    if args.append: # If -a or --append was passed
        os.environ['APPEND'] = 'True'
        print("Launcher CLI: APPEND set to True")
    elif 'APPEND' not in os.environ: # If not set by CLI and not by .env or external
        os.environ['APPEND'] = 'False' # Default to False if nothing else specified it
        #print("Launcher: APPEND defaulted to False (no prior setting)")

    if args.language:
        os.environ['FORCE_DETECTED_LANGUAGE_TO'] = args.language
        print(f"Launcher CLI: FORCE_DETECTED_LANGUAGE_TO set to {args.language}")
    # --- End Environment Variable Handling ---


    requirements_url = "https://raw.githubusercontent.com/McCloudS/subgen/main/requirements.txt"
    requirements_file = "requirements.txt"

    if args.install:
        download_from_github(requirements_url, requirements_file)
        install_packages_from_requirements(requirements_file)

    if not os.path.exists(subgen_script_to_run) or args.update or convert_to_bool(os.getenv('UPDATE')):
        print(f"Downloading {subgen_script_to_run} from GitHub branch {branch_name}...")
        download_from_github(f"https://raw.githubusercontent.com/McCloudS/subgen/{branch_name}/subgen.py", subgen_script_to_run)
        print(f"Downloading {language_code_script_to_download} from GitHub branch {branch_name}...")
        download_from_github(f"https://raw.githubusercontent.com/McCloudS/subgen/{branch_name}/language_code.py", language_code_script_to_download)

    else:
        print(f"{subgen_script_to_run} exists and UPDATE is set to False, skipping download.")

    if args.srt:
        # SRT translation mode: takes priority over --file
        import importlib.util
        srt_path = os.path.abspath(args.srt)
        if not os.path.isfile(srt_path):
            print(f"Error: SRT file not found: {srt_path}")
            sys.exit(1)
        target_language = args.srt_to or os.getenv('TRANSLATE_TO', '').strip()
        if not target_language:
            print("Error: target language is required. Use --srt-to <language> or set TRANSLATE_TO in subgen.env")
            sys.exit(1)
        source_language = (args.srt_source_language or '').strip()
        if not source_language:
            inferred_source_language = infer_source_language_from_srt_path(srt_path)
            if inferred_source_language:
                source_language = inferred_source_language
                print(f"Inferred --srt-source-language from subtitle filename: {source_language}")
        if not source_language:
            print("Error: --srt-source-language is required in --srt mode when source language cannot be inferred from subtitle filename")
            sys.exit(1)
        print(f"Translating SRT: {srt_path}  ->  {target_language}")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from translate import configure_translation, translate_srt_file_to_bilingual
        model_id = os.getenv('TRANSLATEGEMMA_MODEL', 'google/translategemma-4b-it')
        model_location = os.getenv('MODEL_PATH', './models')
        batch_size = max(1, int(os.getenv('TRANSLATE_BATCH_SIZE', 4)))
        configure_translation(
            model_id=model_id,
            model_location=model_location,
            batch_size=batch_size,
        )
        output_path = translate_srt_file_to_bilingual(
            input_srt_path=srt_path,
            target_language=target_language,
            source_language=source_language,
        )
        print(f"Bilingual SRT written to: {output_path}")

    elif args.file:
        # Direct pipeline mode: import subgen as a module and call gen_subtitles directly.
        # This skips server startup — uvicorn is only started when subgen.py is run as __main__.
        import importlib.util
        file_path = os.path.abspath(args.file)
        if not os.path.isfile(file_path):
            print(f"Error: file not found: {file_path}")
            sys.exit(1)
        print(f"Loading {subgen_script_to_run} as module...")
        spec = importlib.util.spec_from_file_location("subgen", subgen_script_to_run)
        subgen_mod = importlib.util.module_from_spec(spec)
        sys.modules["subgen"] = subgen_mod
        spec.loader.exec_module(subgen_mod)
        transcription_type = args.type or os.getenv('TRANSCRIBE_OR_TRANSLATE', 'transcribe').lower()
        force_language = subgen_mod.LanguageCode.NONE
        if args.language:
            force_language = subgen_mod.LanguageCode.from_string(args.language)
            if not force_language:
                print(f"Error: invalid --language value '{args.language}'")
                sys.exit(1)
        print(f"Running pipeline on: {file_path}  (type={transcription_type}, language={args.language or 'auto'})")
        subgen_mod.gen_subtitles(file_path, transcription_type, force_language)

    elif not args.exit_early:
        #print(f"DEBUG environment variable for subgen.py: {os.getenv('DEBUG')}")
        #print(f"APPEND environment variable for subgen.py: {os.getenv('APPEND')}")
        print(f'Launching {subgen_script_to_run}')
        try:
            subprocess.run([python_cmd, subgen_script_to_run], check=True)
            # subprocess.run(["uv", 'run', subgen_script_to_run], check=True)
        except FileNotFoundError:
            print(f"Error: Could not find {subgen_script_to_run}. Make sure it was downloaded correctly.")
        except subprocess.CalledProcessError as e:
            print(f"Error running {subgen_script_to_run}: {e}")

    else:
        print("Not running subgen.py: -x or --exit-early set")

if __name__ == "__main__":
    main()
