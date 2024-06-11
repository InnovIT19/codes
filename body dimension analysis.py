import cv2
import os
import numpy as np

# Step 1: Extract Snapshots from Video
def extract_snapshots(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 30 == 0:  # Capture one frame every second (assuming 30 fps)
            snapshot_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(snapshot_path, frame)
        frame_count += 1
    cap.release()

# Usage
video_path = 'C:/Users/DELL/Desktop/Rashmi/Project/1st/v2.mp4'
output_folder = 'C:/Users/DELL/Desktop/Rashmi/Project/1st/snapshots'

extract_snapshots(video_path, output_folder)

# Step 2: Extract Body Dimensions from Snapshot
def extract_body_dimensions(snapshot_path):
    dimensions = {
        'height': np.random.uniform(150, 200),  # in cm
        'shoulder_width': np.random.uniform(35, 55),  # in cm
        'waist_circumference': np.random.uniform(25, 50),  # in cm
        'hip_circumference': np.random.uniform(20, 50)  # in cm
    }
    return dimensions

snapshot_path = os.path.join(output_folder, 'frame_0.jpg')
dimensions = extract_body_dimensions(snapshot_path)
print(f"Extracted Dimensions: {dimensions}")

# Step 3: Define Reinforcement Learning Model
class ReinforcementLearningModel:
    def __init__(self):
        self.q_table = {}
        self.actions = ['hourglass', 'inverted_triangle', 'rectangle', 'pear']
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1

    def _choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.choice(self.actions)
        else:
            action = max(self.q_table.get(state, {}), key=self.q_table.get(state, {}).get, default=np.random.choice(self.actions))
        return action

    def _learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}

        predict = self.q_table[state][action]
        target = reward + self.discount_factor * max(self.q_table[next_state].values())
        self.q_table[state][action] += self.learning_rate * (target - predict)

    def train(self, dimensions, actual_shape):
        state = tuple(dimensions.values())
        action = self._choose_action(state)
        reward = 1 if action == actual_shape else -1
        self._learn(state, action, reward, state)

    def predict(self, dimensions):
        state = tuple(dimensions.values())
        return self._choose_action(state)

# Step 4: Train and Test the Model
model = ReinforcementLearningModel()

for _ in range(1000):
    dummy_dimensions = {
        'height': np.random.uniform(150, 200),  # in cm
        'shoulder_width': np.random.uniform(35, 55),  # in cm
        'waist_circumference': np.random.uniform(25, 50),  # in cm
        'hip_circumference': np.random.uniform(20, 50)  # in cm
    }
    actual_shape = np.random.choice(['hourglass', 'inverted_triangle', 'rectangle', 'pear'])
    model.train(dummy_dimensions, actual_shape)

# Predict body shape for extracted dimensions
predicted_shape = model.predict(dimensions)
print(f"Predicted body shape: {predicted_shape}")
