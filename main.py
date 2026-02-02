import sys
import os
import subprocess

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_script(script_name):
    script_path = os.path.join("src", script_name)
    if not os.path.exists(script_path):
        print(f"[ERROR] File not found: {script_path}")
        input("Press Enter...")
        return

    try:
        # Run the script and wait for it to finish
        subprocess.run([sys.executable, script_path], check=True)
    except Exception as e:
        print(f"\n[Error] Script stopped: {e}")
    
    input("\nPress Enter to return to menu...")

def main():
    while True:
        clear_screen()
        print("LOCUS CONTROL PANEL\n")
        print("1. Enroll")
        print("2. Insight")
        print("3. Verify")
        print("4. Exit")
        
        choice = input("\nSelect option: ")

        if choice == '1':
            run_script("enroll.py")
        elif choice == '2':
            run_script("insight.py")
        elif choice == '3':
            run_script("verify.py")
        elif choice == '4':
            sys.exit()

if __name__ == "__main__":
    main()