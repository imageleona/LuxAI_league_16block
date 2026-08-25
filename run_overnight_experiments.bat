@echo off
REM ===========================================================================
REM Overnight experiment: does either new mechanism help on 12x12?
REM
REM   1. exp_baseline         CONTROL - plain PFSP
REM   2. exp_winrate_admit    + win-rate-gated pool admission (only difference)
REM   3. exp_research_reward  + research-timing reward        (only difference)
REM
REM Each arm differs from the control by exactly one setting, so a difference is
REM attributable. Sequential, never parallel: this GPU fits one run, and three
REM concurrent runs previously saturated its 16 GB and slowed each ~30x.
REM
REM Budget: 500k steps each at ~77 SPS (12x12) = ~1.8 h per arm, ~5.5 h total.
REM
REM NB 1: this file MUST live at the repository root. "%~dp0" is the directory
REM        of the script, and every path below is relative to it. The scripts in
REM        bat_files/ are broken for exactly this reason - they were moved out of
REM        the root but still call "python run_monobeast.py".
REM NB 2: "call" before conda activate is REQUIRED. conda is a .bat, and without
REM        "call" control transfers and never returns - the script would exit on
REM        line 1 having run nothing.
REM NB 3: keep each python invocation on ONE line. Caret (^) continuations
REM        combined with an output redirect silently produce a command that
REM        starts and then does nothing.
REM ===========================================================================

setlocal
cd /d "%~dp0"

set ENVDIR=C:\Users\_s2111724\2026_summer_camp\LuxAI\lux_env37
set STEPS=500000
set LOGDIR=%~dp0league_run_logs

REM wandb is off by default. A wandb filestream 404 previously hung a run for 20
REM minutes without training a single step, and this is unattended. Everything
REM needed is written to disk anyway: outputs\<date>\<time>\league\outcomes.jsonl,
REM state.json and the run log. Set to False to re-enable.
set DISABLE_WANDB=True

call conda activate %ENVDIR%
if errorlevel 1 (
    echo [ERROR] conda activate failed - aborting.
    exit /b 1
)

REM Fail now rather than after hours of nothing happening.
python -c "import torch, wandb, omegaconf, kaggle_environments" 2>nul
if errorlevel 1 (
    echo [ERROR] wrong interpreter or missing dependencies - aborting.
    python -c "import sys; print('python is:', sys.executable)"
    exit /b 1
)

REM Three arms x ~2.5 GB of checkpoints and snapshots. Bail early rather than
REM dying half way through arm 3 with a truncated checkpoint.
for /f "tokens=3" %%a in ('dir /-c "%~dp0" ^| find "bytes free"') do set FREE=%%a
echo Free disk: %FREE% bytes  (need roughly 10,000,000,000)

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo ===========================================================================
echo [%date% %time%] RUN 1/3  exp_baseline          (CONTROL)
echo ===========================================================================
python run_monobeast.py --config-name exp_baseline total_steps=%STEPS% disable_wandb=%DISABLE_WANDB% > "%LOGDIR%\exp_baseline.log" 2>&1
echo [%date% %time%] exp_baseline exited with code %errorlevel%

echo ===========================================================================
echo [%date% %time%] RUN 2/3  exp_winrate_admit     (+ gated admission)
echo ===========================================================================
python run_monobeast.py --config-name exp_winrate_admit total_steps=%STEPS% disable_wandb=%DISABLE_WANDB% > "%LOGDIR%\exp_winrate_admit.log" 2>&1
echo [%date% %time%] exp_winrate_admit exited with code %errorlevel%

echo ===========================================================================
echo [%date% %time%] RUN 3/3  exp_research_reward   (+ timing reward)
echo ===========================================================================
python run_monobeast.py --config-name exp_research_reward total_steps=%STEPS% disable_wandb=%DISABLE_WANDB% > "%LOGDIR%\exp_research_reward.log" 2>&1
echo [%date% %time%] exp_research_reward exited with code %errorlevel%

echo.
echo [%date% %time%] ALL RUNS COMPLETE.
echo Logs:    %LOGDIR%
echo Results: outputs\%date:~-2%-%date:~4,2%-%date:~7,2%\  (per-run league\outcomes.jsonl)
endlocal
