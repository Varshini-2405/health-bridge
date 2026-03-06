import subprocess
import time
import sys
import webbrowser

def run_project():
    print("🚀 Starting HealthBridge Services...")
    
    # Check if models exist, if not, train them
    import os
    if not (os.path.exists("ai-service/models/risk_model.pkl") and os.path.exists("ai-service/models/disease_model.pkl")):
        print("\n[!] Models not found! Generating data and training now...")
        subprocess.run([sys.executable, "train_model.py"], cwd="ai-service")
        print("[!] Training complete.\n")

    # 1. Start AI-Service (FastAPI) on Port 8000
    print("\n[1/3] Starting AI Intelligence Layer...")
    ai_process = subprocess.Popen(
        [sys.executable, "main.py"], 
        cwd="ai-service",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Give AI service a second to start
    time.sleep(2)
    
    # 2. Start Backend (Flask) on Port 5000
    print("[2/3] Starting Backend Server...")
    backend_process = subprocess.Popen(
        [sys.executable, "app.py"], 
        cwd="server",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Give backend a second to start
    time.sleep(3)
    
    # 3. Open Browser
    print("[3/3] Opening Patient Portal...")
    webbrowser.open("http://localhost:5000")
    
    print("\n✅ System is now running!")
    print("--------------------------------------------------")
    print("Patient Portal: http://localhost:5000")
    print("Admin Panel:   http://localhost:5000/admin")
    print("--------------------------------------------------")
    print("Press Ctrl+C in this terminal to stop both servers.")
    
    try:
        # Keep the script running to monitor processes
        while True:
            # Check if processes are still alive
            if ai_process.poll() is not None:
                print("⚠️ AI Service has stopped unexpectedly.")
                break
            if backend_process.poll() is not None:
                print("⚠️ Backend Server has stopped unexpectedly.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
    finally:
        ai_process.terminate()
        backend_process.terminate()
        print("Done.")

if __name__ == "__main__":
    run_project()
