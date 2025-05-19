import gym
from gym import spaces
import numpy as np
import carla
import torch
import torchvision.models as models
from PIL import Image
import random


risk_aversion_lambda = 0 # Example risk aversion parameter 

'''for risk_aversion_lambda=0 i have trained the model by making reward/10.0 in the step function in line 149 and for 
risk_aversion_lambdas rather than 0 i have trained the model by making just reward in the step function in line 149'''


class ResNetProcessor:
    def __init__(self):
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.eval()

    def process(self, image):
        image = Image.fromarray(image).resize((224, 224))
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = image.transpose(2, 0, 1)
        image_tensor = torch.tensor(image[np.newaxis, ...], dtype=torch.float32)
        with torch.no_grad():
            features = self.resnet(image_tensor).flatten().numpy()
        return features

class CarlaEnv(gym.Env):
    def __init__(self, risk_aversion_lambda=risk_aversion_lambda):
        super(CarlaEnv, self).__init__()

        # Risk parameters  
        self.risk_aversion_lambda = risk_aversion_lambda
        self.reward_history = []
        self.reward_scaling = 5000.0  # From risk.py implementation

        # CARLA connection
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town01')
        self.blueprint_library = self.world.get_blueprint_library()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1, 0.15]), 
            high=np.array([1, 1]), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(2052,), 
            dtype=np.float32
        )
        
        # Initialize actors and components
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.image_processor = ResNetProcessor()
        self.current_step = 0
        self.max_steps = 5000
        self.reset()

    def reset(self):
        self._destroy_actors()
        
        # Spawn vehicle
        vehicle_bp = self.blueprint_library.filter('model3')[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Attach camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera = self.world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=2.4)),
            attach_to=self.vehicle)
        self.camera.listen(lambda image: self._process_camera(image))
        
        # Attach collision sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._handle_collision(event))
        
        # Initialize waypoints
        self.current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        self.target_waypoint = self.current_waypoint.next(5.0)[0]
        
        # Reset state variables
        self.current_image = None
        self.collision_occurred = False
        self.current_step = 0
        self.reward_history = []
        return self._get_obs()

    def _process_camera(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
        self.current_image = array

    def _handle_collision(self, event):
        self.collision_occurred = True

    def _get_obs(self):
        features = self.image_processor.process(self.current_image) if self.current_image is not None else np.zeros(2048)
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)
        throttle = self.vehicle.get_control().throttle
        
        # Relative position to target waypoint
        vehicle_loc = self.vehicle.get_location()
        target_loc = self.target_waypoint.transform.location
        rel_position = np.array([
            target_loc.x - vehicle_loc.x,
            target_loc.y - vehicle_loc.y
        ])
        return np.concatenate([features, [speed, throttle], rel_position])

    def step(self, action):
        self.current_step += 1
        control = carla.VehicleControl()
        
        # Apply action with clipping
        control.steer = float(np.clip(action[0], -1.0, 1.0))
        control.throttle = float(np.clip(max(0.15, action[1]), 0.0, 1.0))
        self.vehicle.apply_control(control)
        self.world.tick()
        
        # Update waypoint
        vehicle_loc = self.vehicle.get_location()
        self.current_waypoint = self.world.get_map().get_waypoint(vehicle_loc)
        try:
            self.target_waypoint = self.current_waypoint.next(5.0)[0]
        except (IndexError, AttributeError):
            self.target_waypoint = self.current_waypoint
        
        # Get observation and calculate reward
        obs = self._get_obs()
        reward, done, termination_reason = self._calculate_reward()
        return obs, reward/10.0, done, {"termination_reason": termination_reason}
    
    def _get_nearest_vehicle_distance(self):
        """Get distance to the nearest vehicle."""
        vehicle_loc = self.vehicle.get_location()
        vehicles = self.world.get_actors().filter('vehicle.*')
        distances = [vehicle_loc.distance(v.get_location()) for v in vehicles if v.id != self.vehicle.id]
        return min(distances) if distances else np.inf

    def _is_out_of_lane(self):
        """Check if the vehicle is out of its lane."""
        lane_width = self.current_waypoint.lane_width
        vehicle_loc = self.vehicle.get_location()
        lateral_distance = abs(vehicle_loc.y - self.current_waypoint.transform.location.y)
        return lateral_distance > lane_width / 2

    def _is_headway_violation(self):
        """Check if the vehicle is too close to the vehicle in front."""
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2)  # km/h
        headway_distance = self._get_nearest_vehicle_distance()
        return headway_distance < speed / 2  # Minimum safe headway = speed/2 (m)

    def _calculate_reward(self):
        vehicle_loc = self.vehicle.get_location()
        target_loc = self.target_waypoint.transform.location
        
        # Parameters
        collision_severity = 0
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2)
        jerk = self.vehicle.get_control().throttle - self.previous_throttle if hasattr(self, 'previous_throttle') else 0
        steering_angle = self.vehicle.get_control().steer
        distance_to_target = vehicle_loc.distance(target_loc)
        
        # === Reward Components ===
        rsafety, rprogress, rcomfort, rrules = 0, 0, 0, 0
        
        # 1. Safety
        if self.collision_occurred:
            collision_severity = 1 + min(speed / 50, 1)
            rsafety = -300 * collision_severity
        else:
            nearest_vehicle_distance = self._get_nearest_vehicle_distance()
            ttc = nearest_vehicle_distance / (speed / 3.6 + 1e-6) if speed > 0 else np.inf
            rsafety = -5 / ttc if ttc < 2 else 5

        # 2. Progress
        if distance_to_target <= 10.0:
            rprogress = 2.0 * np.exp(-distance_to_target / 10.0)
        else:
            rprogress = -0.5
        rprogress += max(0, speed / 40) * 5

        # 3. Comfort
        rcomfort = -abs(jerk) - 0.1 * abs(steering_angle)

        # 4. Traffic Rules
        if speed > 50: rrules -= 5
        if self._is_out_of_lane(): rrules -= 10
        if self._is_headway_violation(): rrules -= 7

        # === Weighted Total Reward ===
        total_reward = (1.5 * rsafety + 
                       1.0 * rprogress + 
                       0.8 * rcomfort + 
                       1.2 * rrules)
        
        # Store for risk calculation
        self.reward_history.append(total_reward)

        # === Entropic Risk Adjustment ===
        if self.risk_aversion_lambda > 0 and len(self.reward_history) > 0:
            rewards_array = np.array(self.reward_history)
            normalized_rewards = rewards_array / self.reward_scaling
            exp_terms = np.exp(-self.risk_aversion_lambda * normalized_rewards)
            exp_terms = np.clip(exp_terms, 1e-4, 1e4)  # Numerical stability
            mean_exp = np.mean(exp_terms)
            risk_adjusted_reward = - (1 / self.risk_aversion_lambda) * np.log(mean_exp)
        else:
            risk_adjusted_reward = total_reward
 
        # === Termination Conditions ===
        done = (self.collision_occurred or 
                distance_to_target > 10.0 or 
                self.current_step >= self.max_steps)
        
        termination_reason = ""
        if done:
            if self.collision_occurred:
                termination_reason = "collision"
            elif distance_to_target > 10.0:
                termination_reason = "deviation"
            else:
                termination_reason = "max_steps"

        self.previous_throttle = self.vehicle.get_control().throttle

        return risk_adjusted_reward, done, termination_reason

    def _destroy_actors(self):
        if self.vehicle: self.vehicle.destroy()
        if self.camera: self.camera.destroy()
        if self.collision_sensor: self.collision_sensor.destroy()

    def close(self):
        self._destroy_actors()