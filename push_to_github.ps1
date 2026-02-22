# Push this Submission directory to a new GitHub repository.
# Paper: Do LLMs and VLMs Share Neurons for Inference? Evidence and Mechanisms of Cross-Modal Transfer
#
# Prerequisites:
# 1. Create a new empty repository on GitHub (e.g. "Do-LLMs-VLMs-Share-Neurons")
# 2. Set GITHUB_TOKEN env var: $env:GITHUB_TOKEN = "your_pat_token"
# 3. Set GITHUB_USER: $env:GITHUB_USER = "your_username"
# 4. Run: .\push_to_github.ps1

param(
    [string]$RepoName = "Do-LLMs-VLMs-Share-Neurons",
    [string]$Token = $env:GITHUB_TOKEN,
    [string]$User = $env:GITHUB_USER
)

if (-not $Token) {
    Write-Error "Set GITHUB_TOKEN environment variable or pass -Token"
    exit 1
}
if (-not $User) {
    Write-Error "Set GITHUB_USER environment variable or pass -User"
    exit 1
}

$RemoteUrl = "https://${User}:${Token}@github.com/${User}/${RepoName}.git"
Write-Host "Initializing git and pushing to ${User}/${RepoName}..."

if (-not (Test-Path .git)) {
    git init
}
git add -A
git status
git commit -m "Initial commit: Neuron merging for cross-modal transfer" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Nothing to commit or already committed."
}
git branch -M main
git remote remove origin 2>$null
git remote add origin $RemoteUrl
git push -u origin main
Write-Host "Done."
