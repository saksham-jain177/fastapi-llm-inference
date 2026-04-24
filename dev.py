import subprocess
import os
import sys
import time
import requests

def check_infra():
    """Start and verify Docker infrastructure."""
    print("🚀 Starting Infrastructure (Docker Compose)...")
    try:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
    except FileNotFoundError:
        print("❌ docker-compose not found. Please install Docker Desktop.")
        sys.exit(1)
    
    # Wait for Prometheus
    print("⏳ Waiting for Prometheus to be healthy...")
    for _ in range(15):
        try:
            resp = requests.get("http://localhost:9090/-/healthy")
            if resp.status_code == 200:
                print("✅ Prometheus is ready.")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("⚠️ Prometheus health check timed out. Continuing anyway...")

def check_gpu():
    """Verify GPU availability on the host."""
    print("🔍 Checking GPU availability...")
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️ No GPU detected. Local inference will be slow (CPU-only).")
            return False
    except ImportError:
        print("⚠️ torch not installed. Did you run 'uv pip sync uv.lock'?")
        return False

def start_backend():
    """Start the FastAPI backend on the host."""
    print("🛰️ Starting Backend (Uvicorn)...")
    # Ensure logs reach console
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    return subprocess.Popen(
        ["uvicorn", "app.main:app", "--reload", "--port", "8000"],
        env=env
    )

def start_frontend():
    """Start the React frontend on the host."""
    print("⚛️ Starting Frontend (Vite)...")
    
    # Auto-install dependencies if missing
    frontend_dir = os.path.join(os.getcwd(), "frontend")
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("📦 First run detected for frontend. Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, shell=True, check=True)
        
    # Use shell=True for Windows npm resolution
    return subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="frontend",
        shell=True
    )

def main():
    backend_proc = None
    frontend_proc = None
    
    try:
        check_infra()
        check_gpu()
        
        backend_proc = start_backend()
        frontend_proc = start_frontend()
        
        print("\n" + "="*40)
        print("✨ STACK ORCHESTRATED SUCCESSFULLY")
        print("="*40)
        print("🔗 Backend API: http://localhost:8000")
        print("🔗 Frontend UI: http://localhost:5173")
        print("🔗 Prometheus:  http://localhost:9090")
        print("🔗 Grafana:     http://localhost:3000 (admin/admin)")
        print("="*40)
        print("\nPress Ctrl+C to teardown stack...")
        
        # Keep alive and monitor processes
        while True:
            if backend_proc.poll() is not None:
                print("❌ Backend process died. Exiting.")
                break
            if frontend_proc.poll() is not None:
                print("❌ Frontend process died. Exiting.")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Graceful teardown initiated...")
    finally:
        if backend_proc:
            backend_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()
        print("🐳 Stopping Docker infrastructure...")
        subprocess.run(["docker-compose", "stop"])
        print("👋 Stack stopped.")

if __name__ == "__main__":
    main()
