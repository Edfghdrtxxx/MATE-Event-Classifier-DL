@echo off
REM ========================================
REM 修复 Git 推送冲突
REM ========================================

echo ========================================
echo 修复 Git 推送冲突
echo ========================================
echo.

cd /d d:\Something\github_repo_forJob\mate_experiment_pdl

echo [1/4] 拉取远程更改...
git pull origin main --allow-unrelated-histories --no-edit
if errorlevel 1 (
    echo 注意：如果出现冲突，我们将使用本地版本
    git merge --abort 2>nul
    git pull origin main --allow-unrelated-histories -X ours --no-edit
)
echo.

echo [2/4] 添加所有文件...
git add .
echo.

echo [3/4] 提交更改...
git commit -m "Fix: Update README URLs and merge remote changes"
if errorlevel 1 (
    echo 注意：可能没有新的更改需要提交
)
echo.

echo [4/4] 推送到 GitHub...
git push -u origin main

if errorlevel 1 (
    echo.
    echo ========================================
    echo 推送失败！
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✓ 成功推送到 GitHub！
echo ========================================
echo.
echo 你的项目地址：
echo https://github.com/Edfghdrtxxx/MATE-Event-Classifier-DL
echo.
echo 现在可以访问上面的地址查看你的项目了！
echo ========================================
pause
