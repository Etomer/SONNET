import pyroomacoustics as pra
import numpy as np
import configs.config_gen as config_gen

class QuadraticBezierCurve():
    
    def __init__(self, room_size, max_time, max_speed):
        """
        Generates a Quadratic Bezier Curve inside a define room.

        Inputs:
        ---------------
            room_size - np.array, shape=[3], room is assumed to be axis aligned with oposite corners in (0,0,0) and room_size
            max_time - curve will be defined as QBC(t), 0 < t < max_time
            max_speed - curve is guaranteed to be shorter than max_speed*max_time
        """
        control_points = np.random.rand(3,3)*room_size

        max_dist = max_time*max_speed/2.0
        # make sure distance between control points is less than max_dist
        limit_dist = lambda x0,x1 :  x0 + min(np.linalg.norm((x1 - x0)),max_dist)*(x1 - x0)/np.linalg.norm((x1 - x0))
        control_points[1] = limit_dist(control_points[0], control_points[1])
        control_points[2] = limit_dist(control_points[1], control_points[2])

        self.cp = control_points
        self.max_time = max_time
        self.max_speed = max_speed

    def __call__(self, t):
        if t < 0 or t > self.max_time:
            raise Exception("Curve is not defined for t outside of 0 < t < max_time")
        return self.cp[1] + (1 - t/self.max_time)**2*(self.cp[0] - self.cp[1]) + (t/self.max_time)**2*(self.cp[2] - self.cp[1])

def simulate_room(signal):

    # select subpart of signal
    n_simulation_samples = config_gen.recording_len + config_gen.extra_samples_start_for_echo
    start_idx = np.random.randint(signal.shape[0] - n_simulation_samples - 1)
    signal = signal[start_idx:start_idx+n_simulation_samples]

    # generate room
    reflection_coeff = float((np.random.rand(1)*(config_gen.reflection_coeff_max- config_gen.reflection_coeff_min) + config_gen.reflection_coeff_min)[0])
    x,y,z = (config_gen.room_max_size - config_gen.room_min_size)*np.random.rand(3) + config_gen.room_min_size
    random_point_in_room = lambda : np.random.rand(3)*[x,y,z]
    corners = np.array([[0,0], [0,y], [x,y], [x,0]]).T 
    room = pra.ShoeBox([x,y,z], fs=config_gen.fs, max_order=3, materials=pra.Material(reflection_coeff, config_gen.scatter_coeff), ray_tracing=False, air_absorption=True)

    # handle directivity of sender and receiver
    if config_gen.directivity:
        dir_type = pra.directivities.DirectivityPattern.CARDIOID
    else:
        dir_type = pra.directivities.DirectivityPattern.OMNI

    random_dir_obj = lambda : pra.directivities.CardioidFamily(
                orientation=pra.directivities.DirectionVector(azimuth=np.random.rand()*360, colatitude=180*np.random.rand(), degrees=True),
                pattern_enum=dir_type,)

    # add sender
    max_time = n_simulation_samples/config_gen.fs
    curve = QuadraticBezierCurve(np.array([x,y,z]), max_time, config_gen.sound_source_max_speed)
    sender_dir_obj = random_dir_obj() # sender is pointing at a constant direction during movement
    for i in range(config_gen.sound_source_locations_per_recording):
        t = i*max_time/config_gen.sound_source_locations_per_recording
        send_pos = curve(t)
        local_signal = signal[int(i*n_simulation_samples/config_gen.sound_source_locations_per_recording):int((i+1)*n_simulation_samples/config_gen.sound_source_locations_per_recording)]
        room.add_source(send_pos,directivity=sender_dir_obj,signal=local_signal,delay=t)

    # add receivers
    R = np.array(np.stack([random_point_in_room() for i in range(config_gen.n_mics)]).T)
    #for i in range(config_gen.n_mics):
    room.add_microphone(R,directivity=[random_dir_obj() for j in range(config_gen.n_mics)])

    # simulation
    room.image_source_model() # compute image sources for reflections
    room.compute_rir()
    room.simulate()

    # store correct piece of sound (i.e after the extra bit)
    sounds = room.mic_array.signals
    sounds = sounds[:,config_gen.extra_samples_start_for_echo:n_simulation_samples]

    # Compute toas
    toas = np.zeros(config_gen.n_mics)
    t_mid = (config_gen.extra_samples_start_for_echo + config_gen.recording_len/2)/config_gen.fs
    for i in range(config_gen.n_mics):
        toas[i] = np.linalg.norm(R[:,i] - curve(t_mid))
    
    return (sounds, toas)