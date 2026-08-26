param(
    [ValidateSet("Smoke", "Screen", "Confirm")]
    [string]$Mode = "Smoke",
    [string[]]$Stages = @(),
    [string]$OutputRoot = (Join-Path $env:TEMP "luxai_recovery")
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "C:\Users\_s2111724\2026_summer_camp\LuxAI\lux_env37\python.exe"
$envDir = Split-Path -Parent $python
$weightsOnlyDir = Join-Path $repoRoot "league_agents\haruto_16block\lux_ai\rl_agent"
$fullCheckpointDir = "C:/Users/_s2111724/2026_summer_camp/LuxAI/Lux-Design-S1-team-g/development/haruto/haruto_top_16block_40k"
$anchorDirs = @(
    (Join-Path $repoRoot "league_agents\haruto_16block"),
    (Join-Path $repoRoot "internal_testing\hall_of_fame\11-24_12-56-23_062179520_must_research"),
    (Join-Path $repoRoot "internal_testing\hall_of_fame\10-10_11-18-12_28576448"),
    (Join-Path $repoRoot "internal_testing\internal_agents\10-02_11-29-02_20000192"),
    (Join-Path $repoRoot "internal_testing\internal_agents\10-08_17-35-45_20000192")
)

$resolvedRepo = [IO.Path]::GetFullPath($repoRoot).TrimEnd('\')
$resolvedOutput = [IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
if ($resolvedOutput.StartsWith($resolvedRepo + '\', [StringComparison]::OrdinalIgnoreCase) -or
    $resolvedOutput.Equals($resolvedRepo, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Recovery output must be outside the repository; got $resolvedOutput"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment was not found: $python"
}
if (-not (Test-Path -LiteralPath (Join-Path $fullCheckpointDir "40000.pt") -PathType Leaf)) {
    throw "Full Team G checkpoint was not found: $fullCheckpointDir/40000.pt"
}

$steps = switch ($Mode) {
    "Smoke" { 1024 }
    "Screen" { 200000 }
    "Confirm" { 400000 }
}
$evalSeeds = switch ($Mode) {
    "Smoke" { 0 }
    "Screen" { 6 }
    "Confirm" { 12 }
}

$common = @(
    "total_steps=$steps",
    "disable_wandb=True"
)
if ($Mode -eq "Smoke") {
    $common += @(
        "num_actors=1",
        "n_actor_envs=2",
        "batch_size=2",
        "unroll_length=8",
        "checkpoint_freq=9999",
        "league.anchor_eval_enabled=False",
        "league.snapshot_interval_updates=1000000000"
    )
}
elseif ($Mode -eq "Screen") {
    # Evaluate the final policy without competing with the learner for GPU/CPU.
    $common += "league.anchor_eval_enabled=False"
}
else {
    # Confirmation runs retain periodic best-model selection. A larger final
    # evaluation is still run after training finishes.
    $common += @(
        "league.anchor_eval_enabled=True",
        "league.anchor_eval_n_seeds=3",
        "league.anchor_eval_every_n_games=200"
    )
}

$stageDefinitions = [ordered]@{
    "01_lr_schedule" = @(
        "name=recovery_01_lr_schedule",
        "load_dir=$($weightsOnlyDir.Replace('\','/'))",
        "checkpoint_file=40000_weights.pt",
        "load_optimizer_state=False",
        "entropy_cost=2e-6",
        "lmb=0.9",
        "use_teacher=False",
        "teacher_kl_cost=0.0"
    )
    "02_optimizer" = @(
        "name=recovery_02_optimizer",
        "load_dir=$fullCheckpointDir",
        "checkpoint_file=40000.pt",
        "load_optimizer_state=True",
        "entropy_cost=2e-6",
        "lmb=0.9",
        "use_teacher=False",
        "teacher_kl_cost=0.0"
    )
    "03_entropy" = @(
        "name=recovery_03_entropy",
        "lmb=0.9",
        "use_teacher=False",
        "teacher_kl_cost=0.0"
    )
    "04_lambda" = @(
        "name=recovery_04_lambda",
        "use_teacher=False",
        "teacher_kl_cost=0.0"
    )
    "05_teacher" = @(
        "name=recovery_05_teacher"
    )
    "06_raw_greedy" = @(
        "name=recovery_06_raw_greedy",
        "league.opponent_sample=False"
    )
    "07_deployed" = @(
        "name=recovery_07_deployed",
        "league.opponent_sample=False",
        "league.opponent_inference_mode=deployed"
    )
    "08_research_reward" = @(
        "name=recovery_08_research_reward",
        "league.opponent_sample=False",
        "league.opponent_inference_mode=deployed",
        "reward_space=GameResultWithResearchTiming",
        "+reward_space_kwargs.research_timing=1.0",
        "+reward_space_kwargs.coal_turn=65",
        "+reward_space_kwargs.coal_fast_turn=60",
        "+reward_space_kwargs.uranium_turn=165",
        "+reward_space_kwargs.uranium_fast_turn=160"
    )
    "09_safeguards" = @(
        "name=recovery_09_safeguards",
        "league.opponent_sample=False",
        "league.opponent_inference_mode=deployed",
        "league.anchor_eval_promote_best=True",
        "league.snapshot_interval_updates=1000000000"
    )
}

$Stages = @($Stages | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$selected = if ($Stages.Count -eq 0) { @($stageDefinitions.Keys) } else { $Stages }
foreach ($stage in $selected) {
    if (-not $stageDefinitions.Contains($stage)) {
        throw "Unknown stage '$stage'. Valid stages: $($stageDefinitions.Keys -join ', ')"
    }
}

New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null
$env:LUX_RECOVERY_OUTPUT_ROOT = $resolvedOutput.Replace('\','/')
$env:PYTHONDONTWRITEBYTECODE = "1"
# A direct call to a conda environment's python.exe does not initialize its DLL
# search path on Windows. Without Library\bin, importing ssl (via wandb) fails.
$env:PATH = "$envDir;$envDir\Library\bin;$envDir\Scripts;$env:PATH"

Push-Location $repoRoot
try {
    if ($Mode -ne "Smoke") {
        # Evaluate the immutable starting model once on the same paired schedule
        # used after training. Reuse the result on subsequent single-stage runs.
        $baselineOutput = Join-Path $resolvedOutput "baseline_eval_$($evalSeeds)_seeds"
        $baselineResults = Join-Path $baselineOutput "anchor_eval.txt"
        if (Test-Path -LiteralPath $baselineResults -PathType Leaf) {
            Write-Host "[$(Get-Date -Format s)] Reusing paired baseline: $baselineResults"
        }
        else {
            Write-Host "[$(Get-Date -Format s)] Evaluating immutable baseline with $evalSeeds paired seeds"
            $baselineArguments = @(
                "-m", "evaluation.anchor_eval",
                "--candidate", $anchorDirs[0],
                "--anchors"
            )
            $baselineArguments += $anchorDirs
            $baselineArguments += @(
                "--n-seeds", $evalSeeds,
                "--map-sizes", 12, 16, 24, 32,
                "--parallel-games", 8,
                "--device", "cuda:0",
                "--output", $baselineOutput
            )
            & $python @baselineArguments
            if ($LASTEXITCODE -ne 0) {
                throw "Paired baseline evaluation failed with exit code $LASTEXITCODE"
            }
            if (-not (Test-Path -LiteralPath $baselineResults -PathType Leaf)) {
                throw "Paired baseline evaluation did not produce $baselineResults"
            }
        }
    }

    foreach ($stage in $selected) {
        Write-Host "[$(Get-Date -Format s)] Starting $stage ($Mode, $steps steps)"
        $stamp = (Get-Date -Format "yyyy-MM-dd_HH-mm-ss") + "_" + [guid]::NewGuid().ToString("N").Substring(0, 8)
        $runDir = Join-Path (Join-Path $resolvedOutput $stage) $stamp
        $arguments = @("run_monobeast.py", "--config-name", "recovery_16block")
        $arguments += $common
        $arguments += $stageDefinitions[$stage]
        $arguments += "hydra.run.dir='$($runDir.Replace('\','/'))'"
        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Stage $stage failed with exit code $LASTEXITCODE"
        }
        Write-Host "[$(Get-Date -Format s)] Completed $stage"
        if ($Mode -ne "Smoke") {
            $candidate = Join-Path $runDir "final_agent"
            if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
                throw "Stage $stage did not export final_agent at $candidate"
            }
            $evalOutput = Join-Path $runDir "post_eval"
            Write-Host "[$(Get-Date -Format s)] Evaluating $stage with $evalSeeds paired seeds"
            $evalArguments = @(
                "-m", "evaluation.anchor_eval",
                "--candidate", $candidate,
                "--anchors"
            )
            $evalArguments += $anchorDirs
            $evalArguments += @(
                "--n-seeds", $evalSeeds,
                "--map-sizes", 12, 16, 24, 32,
                "--parallel-games", 8,
                "--device", "cuda:0",
                "--output", $evalOutput
            )
            & $python @evalArguments
            if ($LASTEXITCODE -ne 0) {
                throw "Post-training evaluation for $stage failed with exit code $LASTEXITCODE"
            }
        }
    }
}
finally {
    Pop-Location
}
