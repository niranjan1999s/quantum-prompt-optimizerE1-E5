import threading
import json
from pathlib import Path
import customtkinter as ctk
import sys

# Ensure src is in path to import training module
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root / "src"))

from training.trainer import TrainConfig, train_classification

ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Quantum Prompt Optimizer (Desktop)")
        self.geometry("800x600")

        # Configure grid layout
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(9, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Experiment Config", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Qubits Slider
        self.label_qubits = ctk.CTkLabel(self.sidebar_frame, text="Number of Qubits:")
        self.label_qubits.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.slider_qubits = ctk.CTkSlider(self.sidebar_frame, from_=2, to=8, number_of_steps=6)
        self.slider_qubits.set(4)
        self.slider_qubits.grid(row=2, column=0, padx=20, pady=(0, 10))

        # Quantum Layers Slider
        self.label_layers = ctk.CTkLabel(self.sidebar_frame, text="Quantum Layers:")
        self.label_layers.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.slider_layers = ctk.CTkSlider(self.sidebar_frame, from_=5, to=40, number_of_steps=35)
        self.slider_layers.set(20)
        self.slider_layers.grid(row=4, column=0, padx=20, pady=(0, 10))

        # Epochs Slider
        self.label_epochs = ctk.CTkLabel(self.sidebar_frame, text="Epochs:")
        self.label_epochs.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.slider_epochs = ctk.CTkSlider(self.sidebar_frame, from_=1, to=50, number_of_steps=49)
        self.slider_epochs.set(5)
        self.slider_epochs.grid(row=6, column=0, padx=20, pady=(0, 10))

        # Run Button
        self.sidebar_button_run = ctk.CTkButton(self.sidebar_frame, text="Run Experiment", command=self.run_experiment_thread, fg_color="green", hover_color="darkgreen")
        self.sidebar_button_run.grid(row=8, column=0, padx=20, pady=20)

        # Main Panel
        self.textbox = ctk.CTkTextbox(self, width=250)
        self.textbox.grid(row=0, column=1, rowspan=4, padx=20, pady=20, sticky="nsew")
        self.textbox.insert("0.0", "Welcome to the Quantum Protocol Desktop Interface.\nConfigure your parameters on the left and click Run Experiment.\n\n")
        self.textbox.configure(state="disabled")

        import sys
        
        class TextboxStream:
            def __init__(self, textbox):
                self.textbox = textbox
            def write(self, msg):
                self.textbox.configure(state="normal")
                self.textbox.insert("end", msg)
                self.textbox.see("end")
                self.textbox.configure(state="disabled")
            def flush(self):
                pass
                
        # Redirect stdout to textbox securely
        self.original_stdout = sys.stdout
        sys.stdout = TextboxStream(self.textbox)

    def run_experiment_thread(self):
        self.sidebar_button_run.configure(state="disabled", text="Running...")
        thread = threading.Thread(target=self.run_experiment)
        thread.start()

    def run_experiment(self):
        cfg = TrainConfig(
            dataset_name="glue",
            dataset_config="sst2",
            model_name="gpt2",
            epochs=int(self.slider_epochs.get()),
            lr=0.01,
            n_qubits=int(self.slider_qubits.get()),
            n_layers=int(self.slider_layers.get()),
            embed_dim=768,
            seed=0,
            batch_size=16,
            print_every=1,
            checkpoint_every=10,
            plot=False
        )
        try:
            print(f"Starting execution with {cfg.n_qubits} qubits, {cfg.n_layers} layers up to {cfg.epochs} epochs...\n")
            result = train_classification(cfg)
            
            print("\n----- EXPERIMENT COMPLETE -----")
            print(f"Final Quantum Accuracy: {result['quantum_acc_history'][-1]:.3f}")
            print(f"Final Classical David Accuracy: {result['david_acc_history'][-1]:.3f}")
            print(f"Final Classical Goliath Accuracy: {result['classical_acc_history'][-1]:.3f}")
            
        except Exception as e:
            print(f"\n[ERROR] execution failed: {str(e)}")
        finally:
            self.sidebar_button_run.configure(state="normal", text="Run Experiment")

    def on_closing(self):
        sys.stdout = self.original_stdout
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
