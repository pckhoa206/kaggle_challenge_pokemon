import os
import random
import glob
import shutil
import torch as th
from env import PokemonTCGEnv
from model import PokemonFeatureExtractor
from train import OpponentPolicy, evaluate_agent, export_to_onnx
from sb3_contrib import MaskablePPO

LEAGUE_DIR = "league_archive"
RING_BUFFER_DIR = os.path.join(LEAGUE_DIR, "ring_buffer")
MILESTONE_DIR = os.path.join(LEAGUE_DIR, "milestones")
FOUNDATION_DIR = os.path.join(LEAGUE_DIR, "foundation")

# Constants for Deck Generation
TRAINERS = [1145, 1152, 1227, 1235, 1262, 1205]
def build_deck(mon_ids, energy_id, ratio):
    num_mons, num_trainers, num_energy = ratio
    deck = []
    mon_pool = mon_ids * 4
    trainer_pool = TRAINERS * 4
    random.shuffle(mon_pool)
    random.shuffle(trainer_pool)
    
    deck.extend(mon_pool[:num_mons])
    deck.extend(trainer_pool[:num_trainers])
    while len(deck) < 60:
        deck.append(energy_id)
    return deck[:60]

def init_league():
    for d in [LEAGUE_DIR, RING_BUFFER_DIR, MILESTONE_DIR, FOUNDATION_DIR]:
        os.makedirs(d, exist_ok=True)
        
    # Foundation 1: Grass Combo
    grass_deck = build_deck([25, 27, 28, 45, 1], 1, (8, 21, 31))
    grass_path = os.path.join(FOUNDATION_DIR, "grass_combo")
    if not os.path.exists(f"{grass_path}.zip"):
        env = PokemonTCGEnv(player_deck=grass_deck)
        model = init_model(env)
        model.save(f"{grass_path}.zip")
        with open(f"{grass_path}.csv", "w") as f:
            f.write("\n".join(map(str, grass_deck)) + "\n")
            
    # Foundation 2: Water Control
    water_deck = build_deck([33, 35, 47, 722, 803], 3, (16, 24, 20))
    water_path = os.path.join(FOUNDATION_DIR, "water_control")
    if not os.path.exists(f"{water_path}.zip"):
        env = PokemonTCGEnv(player_deck=water_deck)
        model = init_model(env)
        model.save(f"{water_path}.zip")
        with open(f"{water_path}.csv", "w") as f:
            f.write("\n".join(map(str, water_deck)) + "\n")
            
    # Foundation 3: Heuristic is implicit (no model needed)

def init_model(env):
    policy_kwargs = dict(
        features_extractor_class=PokemonFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])
    )
    return MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=2e-4, n_steps=2048, batch_size=128, verbose=0)

def get_random_ghost(category):
    if category == "recent":
        files = glob.glob(os.path.join(RING_BUFFER_DIR, "*.zip"))
    elif category == "historical":
        files = glob.glob(os.path.join(MILESTONE_DIR, "*.zip")) + glob.glob(os.path.join(FOUNDATION_DIR, "*.zip"))
    
    if not files:
        return None, None
        
    chosen_zip = random.choice(files)
    chosen_csv = chosen_zip.replace(".zip", ".csv")
    if not os.path.exists(chosen_csv):
        return None, None
        
    return chosen_zip, chosen_csv

def manage_ring_buffer(champion_model, champion_deck, epoch):
    save_path = os.path.join(RING_BUFFER_DIR, f"epoch_{epoch}")
    champion_model.save(f"{save_path}.zip")
    with open(f"{save_path}.csv", "w") as f:
        f.write("\n".join(map(str, champion_deck)) + "\n")
        
    # Prune
    files = sorted(glob.glob(os.path.join(RING_BUFFER_DIR, "*.zip")), key=os.path.getctime)
    if len(files) > 20:
        for old_file in files[:-20]:
            os.remove(old_file)
            csv_file = old_file.replace(".zip", ".csv")
            if os.path.exists(csv_file):
                os.remove(csv_file)

def save_milestone(champion_model, champion_deck, epoch):
    save_path = os.path.join(MILESTONE_DIR, f"milestone_{epoch}")
    champion_model.save(f"{save_path}.zip")
    with open(f"{save_path}.csv", "w") as f:
        f.write("\n".join(map(str, champion_deck)) + "\n")

import gc

def main():
    print("========================================")
    print("   ALPHAGO LEAGUE TRAINING SYSTEM")
    print("========================================")
    
    init_league()
    
    # Load Champion
    with open("deck.csv", "r") as f:
        champion_deck = [int(line.strip()) for line in f.read().split("\n") if line.strip()][:60]
        
    env = PokemonTCGEnv(player_deck=champion_deck)
    print("Loading Fire_Balanced Champion...")
    if os.path.exists("model_Champion.zip"):
        model = MaskablePPO.load("model_Champion.zip", env=env)
    else:
        print("WARNING: model_Champion.zip not found! Starting fresh.")
        model = init_model(env)
        
    EPOCHS = 32
    STEPS_PER_EPOCH = 250000
    
    print(f"Starting League Training for {EPOCHS} Epochs ({EPOCHS * STEPS_PER_EPOCH} steps)...")
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        # Matchmaking
        opp_policy = OpponentPolicy(mode="rl")
        env.opponent_deck_override = None
        opponent_name = "Self/Heuristic"
        
        # 80/20 Split
        if random.random() < 0.8:
            # 80% Recent Ghost
            ghost_zip, ghost_csv = get_random_ghost("recent")
            if not ghost_zip: # Fallback if ring buffer is empty
                ghost_zip, ghost_csv = get_random_ghost("historical")
        else:
            # 20% Historical/Foundation/Boss
            rand_val = random.random()
            if rand_val < 0.2: # 4% pure heuristic
                opp_policy.mode = "heuristic"
                ghost_zip = "Heuristic Bot"
                ghost_csv = None
            elif rand_val < 0.6: # 8% Tank Boss (to cure Litten Rush)
                opp_policy.mode = "heuristic"
                ghost_zip = "Tank Boss"
                ghost_csv = "opponent_decks/deck_tank_boss.csv"
            else: # 8% Historical Ghosts
                ghost_zip, ghost_csv = get_random_ghost("historical")
                
        if ghost_zip and opp_policy.mode == "rl":
            opp_model = MaskablePPO.load(ghost_zip)
            opp_policy.model = opp_model
            opponent_name = os.path.basename(ghost_zip)
            
            with open(ghost_csv, "r") as f:
                opp_deck = [int(line.strip()) for line in f.read().split("\n") if line.strip()][:60]
            env.opponent_deck_override = opp_deck
            
        env.opponent_policy = opp_policy
        print(f"Matchup: Champion vs {opponent_name}")
        
        # Train
        model.learn(total_timesteps=STEPS_PER_EPOCH)
        
        # Memory Cleanup (Garbage Collection)
        opp_policy.model = None
        if 'opp_model' in locals():
            del opp_model
        gc.collect()
        if th.backends.mps.is_available():
            th.mps.empty_cache()
            
        # Snapshot
        manage_ring_buffer(model, champion_deck, epoch)
        if epoch % 5 == 0:
            save_milestone(model, champion_deck, epoch)
            print(f"Milestone saved for Epoch {epoch}")
            
            # Evaluate against baseline
            eval_env = PokemonTCGEnv(player_deck=champion_deck, opponent_policy=OpponentPolicy(mode="heuristic"))
            wr = evaluate_agent(eval_env, model, eval_games=30)
            print(f"Current Win Rate vs Heuristic: {wr*100:.1f}%")
            
            # Cleanup eval_env
            del eval_env
            gc.collect()

    print("\nLeague Training Complete! Saving final Absolute Champion...")
    model.save("model_Absolute_Champion.zip")
    export_to_onnx(model, "model.onnx")
    print("Done. Submission files updated.")

if __name__ == "__main__":
    main()
