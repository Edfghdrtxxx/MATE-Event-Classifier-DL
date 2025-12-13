@echo off
REM ========================================
REM GitHub 上传自动化脚本
REM ========================================

echo ========================================
echo MATE Event Classifier - GitHub 上传
echo ========================================
echo.

REM 检查是否在正确的目录
if not exist "models\model.py" (
    echo 错误：请在 mate_experiment_pdl 目录下运行此脚本！
    pause
    exit /b 1
)

echo [1/7] 检查 Git 安装...
git --version >nul 2>&1
if errorlevel 1 (
    echo 错误：Git 未安装！
    echo 请访问 https://git-scm.com/download/win 下载安装
    pause
    exit /b 1
)
echo ✓ Git 已安装
echo.

echo [2/7] 初始化 Git 仓库...
if not exist ".git" (
    git init
    echo ✓ Git 仓库已初始化
) else (
    echo ✓ Git 仓库已存在
)
echo.

echo [3/7] 配置 Git 用户信息...
echo 请输入你的 GitHub 用户名（例如：zhiheng-hu）：
set /p USERNAME=
echo 请输入你的邮箱：
set /p EMAIL=

git config user.name "%USERNAME%"
git config user.email "%EMAIL%"
echo ✓ 用户信息已配置
echo.

echo [4/7] 添加所有文件...
git add .
echo ✓ 文件已添加到暂存区
echo.

echo [5/7] 提交更改...
git commit -m "Initial commit: Physics-Informed Deep Learning for MATE Experiment"
if errorlevel 1 (
    echo 注意：可能没有新的更改需要提交
)
echo ✓ 更改已提交
echo.

echo [6/7] 添加远程仓库...
echo 请输入你的 GitHub 仓库地址（例如：https://github.com/zhiheng-hu/MATE-Event-Classifier-DL.git）：
set /p REPO_URL=

git remote remove origin >nul 2>&1
git remote add origin %REPO_URL%
echo ✓ 远程仓库已添加
echo.

echo [7/7] 推送到 GitHub...
git branch -M main
git push -u origin main

if errorlevel 1 (
    echo.
    echo ========================================
    echo 推送失败！可能的原因：
    echo 1. 需要登录 GitHub（会弹出浏览器窗口）
    echo 2. 仓库地址错误
    echo 3. 网络问题
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✓ 成功上传到 GitHub！
echo ========================================
echo.
echo 你的项目地址：
echo %REPO_URL:~0,-4%
echo.
echo 下一步：
echo 1. 访问上面的地址查看你的项目
echo 2. 在简历中添加项目链接
echo 3. 准备面试时的项目介绍
echo ========================================
pause
