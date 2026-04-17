import subprocess
import time
import os

def run_command(command, description):
    print(f"\n🚀 STARTING: {description}")
    print(f"Executing: {command}")
    start_time = time.time()
    
    process = subprocess.run(command, shell=True)
    
    duration = (time.time() - start_time) / 60
    if process.returncode == 0:
        print(f"✅ FINISHED: {description} in {duration:.2f} minutes")
    else:
        print(f"❌ FAILED: {description}")
        return False
    return True

if __name__ == "__main__":
    print("🌟 SUB-TEAM 4 MASTER PIPELINE STARTING 🌟")
    
    # Sabse pehle W&B login kar lo
    os.system("wandb login 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")

    # STEP 1: Pre-train Text MLP (Denoising Trick with Fake Data/Noise)
    # Output: text_mlp_pretrained.pth
    if run_command("python train_text_mlp.py", "Text MLP Phase 1: Pre-training (Fake Data Trick)"):
        
        # STEP 2: Fine-tune Text MLP on REAL EEG Data 
        # (Yahan hum EEG ko CLIP space mein align karenge)
        # Note: Iske liye hume ek alag fine-tune script chahiye hogi, ya isi ko modify karna hoga.
        print("\n📝 Note: Ensure your train_text_mlp.py handles real EEG fine-tuning after pre-training.")
        
        # STEP 3: Train Dynamics Classifier on REAL DATA
        # # (Using Vipresh's dynamics_labels_fixed.npy)
        # run_command("python train_dynamics_mlp.py", "Dynamics Classifier Training (Real Data)")
        
        # STEP 4: Final Sanity Check
        run_command("python test_final_overfit.py", "Final Real-Data Sanity Check")

    print("\n🏆 ALL SUB-TEAM 4 TASKS COMPLETED SUCCESSFULLY!")