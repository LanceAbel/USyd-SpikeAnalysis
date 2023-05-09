
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks, medfilt
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.io.wavfile as wav
import csv
from scipy.stats import sem

plt.gcf().subplots_adjust(bottom=-19.22)



# alternate option without .gcf
# plt.subplots_adjust(bottom=0.15)

# Plot a time-range of the signal and highlight the spikes


#exp_file = r'//Users//user//odrive//OneDrive//Docs Sync//Jobs And Money//Careers//Study//Uni//USyd//2023S1//NEUR3906 Neural Information Processing//Assessments//Prac Reports//Prac 1 - Cockroach Spike Recordings//E1//E1_Light_small_stimulation.wav'
#exp_file = r'//Users//user//odrive//OneDrive//Docs Sync//Jobs And Money//Careers//Study//Uni//USyd//2023S1//NEUR3906 Neural Information Processing//Assessments//Prac Reports//Prac 1 - Cockroach Spike Recordings//E1//E1_Medium_small_stimulation.wav'
#exp_file = r'//Users//user//odrive//OneDrive//Docs Sync//Jobs And Money//Careers//Study//Uni//USyd//2023S1//NEUR3906 Neural Information Processing//Assessments//Prac Reports//Prac 1 - Cockroach Spike Recordings//E1//E1_Strong_small_stimulation.wav'

exp_file = r'//Users//user//odrive//OneDrive//Docs Sync//Jobs And Money//Careers//Study//Uni//USyd//2023S1//NEUR3906 Neural Information Processing//Assessments//Prac Reports//Prac 1 - Cockroach Spike Recordings//E2//E2_Air_1L.wav'
#exp_file = r'//Users//user//odrive//OneDrive//Docs Sync//Jobs And Money//Careers//Study//Uni//USyd//2023S1//NEUR3906 Neural Information Processing//Assessments//Prac Reports//Prac 1 - Cockroach Spike Recordings//E2//E2_Air_2L.wav'
#exp_file = r'//Users//user//odrive//OneDrive//Docs Sync//Jobs And Money//Careers//Study//Uni//USyd//2023S1//NEUR3906 Neural Information Processing//Assessments//Prac Reports//Prac 1 - Cockroach Spike Recordings//E2//E2_Air_3L.wav'

#exp_file = r'//Users//user//odrive//OneDrive//Docs Sync//Jobs And Money//Careers//Study//Uni//USyd//2023S1//NEUR3906 Neural Information Processing//Assessments//Prac Reports//Prac 1 - Cockroach Spike Recordings//E3//E3_LowT_Light_small.wav'
#exp_file = r'//Users//user//odrive//OneDrive//Docs Sync//Jobs And Money//Careers//Study//Uni//USyd//2023S1//NEUR3906 Neural Information Processing//Assessments//Prac Reports//Prac 1 - Cockroach Spike Recordings//E3//E3_LowT_Medium_small.wav'
#exp_file = r'//Users//user//odrive//OneDrive//Docs Sync//Jobs And Money//Careers//Study//Uni//USyd//2023S1//NEUR3906 Neural Information Processing//Assessments//Prac Reports//Prac 1 - Cockroach Spike Recordings//E3//E3_LowT_Strong_small.wav'



class SpikeAnalysis():
    def __init__(self, file_path):

        self.file_path = file_path

        # Spike detection
        self.spike_threshold_sd = 3.5 # if exp_file == r'//Users//user//odrive//OneDrive//Docs Sync//Jobs And Money//Careers//Study//Uni//USyd//2023S1//NEUR3906 Neural Information Processing//Assessments//Prac Reports//Prac 1 - Cockroach Spike Recordings//E1//E1_Strong_small_stimulation.wav' else 4 # standard deviation threshold for spike detection
        self.window_size = 100.0

        self.total_time = 500
        self.num_bins = 20 # int(self.total_time / self.bin_width) # Total self.bin_width * self.num_bins  analysed (in ms)
        self.bin_width = self.total_time / self.num_bins
        

        # Bandpass
        self.low_freq = 1
        self.high_freq = 15000

        self.load_data()
        self.remove_dc()
        self.set_thresholds()

        self.analysis_begin_secs    = 0 # 9.0 # Do not analyse before this (in secs)
        self.analysis_end_secs      = 132 # Do not analyse beyond this (in secs)
        print("Time range selected %s to %s secs"%(str(self.analysis_begin_secs),str(self.analysis_end_secs)))
        # Set the start and end times for the plot
        # Define the time points of the signal
        self.times = np.arange(len(self.signal)) / self.samplerate
        
        #self.signal_to_use = self.signal_dc_filt # self.filtered_signal # self.signal_dc_filt # self.signal_float #  #  # self.signal_float
        #self.select_time_range(start_idx,end_idx)
        #self.select_time_ranges(self.intervals_stimulation)

        self.run_loop()

    def run_loop(self):
        ''' Does signal analysis'''

        # Use v1 for E1-strong
        #self.detect_spikes_v1(threshold = self.spike_threshold_sd, window_size = self.window_size, low_freq=self.low_freq, high_freq=self.high_freq)
        self.detect_spikes_v2()

        print('Number of spikes:', len(self.spike_times))
        #print('Spike times:', self.spike_times)

        #self.plot_spikes()
        self.find_peak_times()
        #self.plot_peak_times()


        self.intervals_use = True
        self.use_stimulation = True # Whether to use period of stimulation or non-stimulation (only applicable if self.intervals_use==True)
        self.filename_intervals   = ""
        self.intervals_namestring = ""
        if self.intervals_use:
            # E2
            self.intervals_to_use = [[5.3,6.5],[10.3,11.5], [15.3,16.5], [20.0,21.2], [25.2,26.4]]  # Intervals in seconds where the stimulation was active
            # E1 - High temp
            #self.intervals_to_use = [[10.4,19.5], [31,39.5], [50.5,59.5], [70.3,79.5], [90.3,99.5], [111,129]] # Intervals in seconds where the stimulation was active
            # E3 - Low temp
            #self.intervals_to_use = [[10.4,19.5], [30.5,39.5], [50.0,59.5], [70.3,79.5], [90.3,99.5], [110.5,129]] 
            self.intervals_namestring = "_stim_only" if self.use_stimulation else "_non_stim_only"
            if not self.use_stimulation:
                # Automatic selection of non-stimulus times from the stimulus times (any other time other than stimulated)
                #self.select_nonstimulation_times()
                # Manual override of non-stimulus times
                #E2
                self.intervals_to_use = [[0,5],[8,10], [12,15], [18,20], [22,25]]  # Intervals in seconds where the stimulation was active
                # E1, E3
                #self.intervals_to_use = [[0,10], [22,30], [42,50], [62,70], [82,90], [102,110]] 
            else:
                print("Intervals to use",self.intervals_to_use )
            # for time_range in self.intervals_to_use:
            #     self.filename_intervals += str(time_range[0]) + "-" + str(time_range[1])

        self.pairs = self.intervals_to_use if self.intervals_use else [(x,x+1) for x in range(0,int(max(self.times)))]
        print("Time range of data %s to %s secs"%(str(min(self.times)),str(max(self.times))))
        


        self.save_spikerate()
        
        #self.plot_STA()

        self.plot_PTSH_auto()
        #self.plot_PSTH_v1()
        ##self.plot_PSTH_v2()

        self.spikerate_stats()
        self.subsequent_firing_rate()


    def load_data(self):
        # Read the .wav file and get the signal
        self.samplerate, self.signal = wavfile.read(self.file_path)
        print("Loaded data")
        print("Sample rate", self.samplerate)


    def remove_dc(self):
        # Find the median of the signal as the baseline
        self.baseline = np.median(self.signal)
        if type(self.signal)!=type([]):
            self.signal_float = self.signal.astype(np.float64)
        else:
            self.signal_float = self.signal[0:]
        self.signal_dc_filt = np.array(self.signal_float, copy=True)
        self.signal_dc_filt -= self.baseline

        self.window_size = 21 # 21 
        self.filtered_signal = sig.medfilt(self.signal_dc_filt, self.window_size)
        self.signal_to_use = self.signal_dc_filt # self.filtered_signal # self.signal_dc_filt # self.signal_float #  #  # self.signal_float



    def set_thresholds(self):
        ## Set the lower and upper thresholds
        # Using absolutes
        # self.low_threshold = 12000
        # self.high_threshold = 40000

        # Using std
        self.low_threshold = self.baseline + self.spike_threshold_sd * np.std(self.signal_dc_filt)
        self.high_threshold = self.baseline + self.spike_threshold_sd*5 * np.std(self.signal_dc_filt)

        print("Low threshold %s high threshold %s"%(str(self.low_threshold),str(self.high_threshold)))


    def select_time_range(self, start_idx, end_idx):
        '''Cuts out times not selected'''
        print("Begin Range", len(self.signal_to_use))
        start_idx = int(self.analysis_begin_secs * self.samplerate)
        end_idx = int(self.analysis_end_secs * self.samplerate)

        self.signal_to_use = self.signal_to_use[start_idx:end_idx]
        self.times = np.arange(len(self.signal_to_use)) / self.samplerate
        print("End Ranges", len(self.signal_to_use),len(self.times))


    def select_time_ranges(self, ranges):
        '''Cuts out times not selected'''

        print("Begin Ranges", len(self.signal_to_use),len(self.times))
        self.new_signal = []
        for time_range in ranges:
            start_idx = int(time_range[0] * self.samplerate)
            end_idx = int(time_range[1] * self.samplerate)
            sig_to_add = self.signal_to_use[start_idx:end_idx]
            #print(time_range, start_idx, end_idx, len(sig_to_add))
            for item in sig_to_add:
                self.new_signal.append(item)

        self.signal_to_use = self.new_signal[0:]
        self.times = np.arange(len(self.signal_to_use)) / self.samplerate
        print("End Ranges", len(self.signal_to_use),len(self.times))

   
  
    def select_nonstimulation_times(self):
        self.intervals_nonstimulation = []
        start_stim = self.intervals_to_use[0][0]

        r = 0
        last_stim = self.intervals_to_use[0][1]  
        #print(start_stim,last_stim,self.intervals_nonstimulation)
        if start_stim > 0: #(10,30)
            self.intervals_nonstimulation.append([0,start_stim])
        else: # (0,4)
            pass
        begin_nonstim = self.intervals_to_use[r][1]
        r+=1

        while r < len(self.intervals_to_use):
            end_nonstim = self.intervals_to_use[r][0]
            self.intervals_nonstimulation.append([begin_nonstim,end_nonstim])
            begin_nonstim = self.intervals_to_use[r][1]
            r+=1

        final_time = max(self.times)
        if last_stim <= final_time:
            self.intervals_nonstimulation.append([begin_nonstim,final_time])

        self.intervals_to_use = self.intervals_nonstimulation[0:]
        print("Non-stimulation intervals", self.intervals_to_use)
        #self.select_time_ranges(self.intervals_to_use)

    def is_in_interval(self, time):
        if self.intervals_use:
            in_interval = False
            for time_start, time_end in self.intervals_to_use:
                if time >= time_start and time <= time_end:
                    in_interval = True
            return in_interval
        else:
            return True

    def generate_buckets(self):
        '''Takes the list of intervals used e.g. 13-15,  33-35 and generates the one second bucket dictionary i.e. {13: [], 14: [], 33: [], 34: []}
        These represent the intervals 13-14, 14-15, 33-34, 34-35'''

        self.one_sec_buckets = {}
        for time_start, time_end in self.pairs:
            all_buckets = range(int(time_start), int(time_end))
            for time_start in all_buckets:
                self.one_sec_buckets[int(time_start)] = 0

    def detect_spikes_v1(self, threshold = 4, window_size = 100.0, low_freq = 1, high_freq = 15000): # self.spike_threshold_sd, window_size = self.window_size, low_freq= self.low_freq, high_freq= self.high_freq):
        analysis_end_secs = self.samplerate
        self.dt = 1 / self.samplerate # bin width in seconds
        #print("dt", self.dt )

        # Apply bandpass filter to extract the frequencies of interest
        nyquist = 0.5 * self.samplerate
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(4, [low, high], btype='band')
        self.filtered_signal = filtfilt(b, a, self.signal_to_use)#self.signal)

        # Compute the rolling median of the signal to estimate the baseline
        #window_size_samples = int(window_size * self.samplerate)
        #self.baseline = medfilt(self.filtered_signal, window_size_samples)
        # Estimate the baseline of the signal using the median
        self.baseline = np.median(self.filtered_signal)

        # Detect spikes that are above a certain threshold above the baseline
        std_dev = np.std(self.filtered_signal - self.baseline)
        threshold *= std_dev
        peaks, _ = find_peaks(self.filtered_signal, height=threshold)

        # Calculate the spike times and durations
        self.spike_times = peaks / self.samplerate
        
        self.spike_durations = np.diff(np.append(-1, peaks)) / self.samplerate
        #print('Spike durations:', self.spike_durations)
 

    def detect_spikes_v2(self):
        # Find the rolling median of the signal

        # # Find the indices of the spike times where the filtered signal crosses the threshold
        #print("Signal to use", self.signal_to_use)
        self.spike_mask = ((self.signal_to_use[:-1] < self.high_threshold) & (self.signal_to_use[1:] > self.high_threshold)) | ((self.signal_to_use[:-1] > self.low_threshold) & (self.signal_to_use[1:] < self.low_threshold))
        #print("Times", self.times)
        self.spike_times = self.times[1:][self.spike_mask]
        #print("Spike times", self.spike_times)

    def plot_spikes(self):

        # Set the start and end times for the plot
        # Find the indices of the time points between analysis_begin_secs and analysis_end_secs seconds
        start_idx = int(self.analysis_begin_secs * self.samplerate)
        end_idx = int(self.analysis_end_secs * self.samplerate)

        # Plot the filtered signal between analysis_begin_secs and analysis_end_secs seconds
        plt.figure(figsize=(10,5))
        plt.title("Signal over time showing spikes")
        plt.plot(self.times[start_idx:end_idx], self.signal_to_use[start_idx:end_idx], 'k-', lw=0.5)


 
    def find_peak_times(self):
        # Highlight the peak of each spike between analysis_begin_secs and analysis_end_secs seconds using a different color or marker
        self.spike_mask_t1t2 = (self.spike_times >= self.analysis_begin_secs) & (self.spike_times <= self.analysis_end_secs)
        self.spike_times_t1t2 = self.spike_times[self.spike_mask_t1t2]
        self.spike_times_to_use = self.spike_times_t1t2
        self.peak_times = []
        #print("Spike times to use in secs", self.spike_times_to_use)
        for t in self.spike_times_to_use:
            t_idx = int(t * self.samplerate)
            window = self.signal_to_use[max(0, t_idx - self.window_size): min(t_idx + self.window_size, len(self.signal_to_use) - 1)]
            peak_idx = np.argmax(window)
            peak_time = (max(0, t_idx - self.window_size) + peak_idx) / self.samplerate
            self.peak_times.append(peak_time)

    def plot_peak_times(self):
        for t in self.spike_times_to_use:
            t_idx = int(t * self.samplerate)
            window = self.signal_to_use[max(0, t_idx - self.window_size): min(t_idx + self.window_size, len(self.signal_to_use) - 1)]
            peak_idx = np.argmax(window)
            peak_time = (max(0, t_idx - self.window_size) + peak_idx) / self.samplerate
            plt.plot([peak_time], [window[peak_idx]], 'ro', ms=4)            
        # Add a legend explaining the red dots are the peak of each spike and the blue line is the STA
        plt.figure()
        plt.legend(['Filtered Signal', 'Spike Peaks'])
        #plt.legend(['Filtered Signal', 'Spike Peaks', 'STA'])
        plt.xlabel('Time (s)')
        plt.ylabel('Signal Amplitude')
        #plt.show()


    def plot_STA(self):

        # Compute the STA - 
        window_size_sta = int(0.01 * self.samplerate)  # 10 ms window
        sta = np.zeros(window_size_sta)
        for t in self.peak_times: # self.spike_times_to_use: # self.spike_times: # Either use self.spike_times for when threshold is breached or peak_times if using the peak of each spike
            t_idx = int(t * self.samplerate)
            if t_idx > window_size_sta:
                sta += self.signal_to_use[t_idx - window_size_sta: t_idx]
        sta /= len(self.spike_times)



        # Plot the STA 
        # Using peak of spike
        sta_times = np.arange(-window_size_sta//2, window_size_sta//2) / self.samplerate 
        # Using start of spike
        #sta_times = np.arange(0, (2*window_size_sta//2)) / self.samplerate
        #plt.plot(sta_times, sta, 'b-', lw=2)
        sta_times = [1000*x for x in sta_times]

        # Plot the STA
        fig, ax = plt.subplots()
        plt.figure()
        plt.plot(sta_times, sta, 'b-', lw=2)
        plt.xlabel('Time (ms) relative to peak of spike')
        plt.ylabel('Signal Value')
        plt.title('Spike-triggered average')
        #plt.show()




    def plot_PTSH_auto(self):
        ## Gerstein and Kiang's method for finding bin size
        # define the maximum number of bins to consider
        max_bins = 100

        # compute the first and second differences of spike times
        dt = np.diff(self.spike_times_to_use)
        d2t = np.diff(dt)


        # compute the range of possible bin sizes
        bin_sizes = np.linspace(dt.min(), dt.max(), max_bins)

        #print("Bin Sizes", bin_sizes)

        # compute the normalized mean square successive difference for each bin size
        nmsd = []
        for bs in bin_sizes:
            counts, _ = np.histogram(self.spike_times_to_use, bins=np.arange(0, self.spike_times_to_use[-1]+bs, bs))
            n = counts.sum()
            if n > 1:
                v = np.var(counts) / np.mean(counts)**2
                nmsd.append((bs / np.mean(counts))**2 * v)

        # find the bin size with the minimum normalized mean square successive difference
        opt_bin_size = bin_sizes[np.argmin(nmsd)]

        # plot the spike times histogram with the optimal bin size
        plt.figure()
        plt.hist(self.spike_times_to_use, bins=np.arange(0, self.spike_times_to_use[-1]+opt_bin_size, opt_bin_size))
        plt.title("Peri-stimulus time histogram")
        if self.intervals_use:
            m = 0
            for j in range(0,len(self.intervals_to_use)):
                i_0, i_1 = self.intervals_to_use[m][0], self.intervals_to_use[m][1]
                plt.axvspan(i_0, i_1, alpha=0.5, color='gray')
                m+=1
        plt.xlabel('Time(s) auto-binned')
        plt.ylabel('Number of spikes/second')
        #plt.show()





    def plot_PSTH_v1(self):
        # Plot PSTH
        # Define the window size and bin size for the PSTH
        window_size = min(140,max(self.spike_times)) # in sec
        bin_size = 2 # in sec
        self.dt = 1 / self.samplerate # bin width in seconds

        # Convert window_size and bin_size to units of samples
        window_samples = int(window_size / self.dt)
        bin_samples = int(bin_size / self.dt)



        # Compute the PSTH
        psth, bins = np.histogram(self.spike_times_to_use, bins=np.arange(0, window_samples+bin_samples, bin_samples) * self.dt)
        psth = psth / (bin_size / 1000) / len(self.spike_times)

        # Plot the PSTH
        fig, ax = plt.subplots()
        ax.bar(bins[:-1], psth, width=bin_size, align='edge')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Spikes per second')
        ax.set_title('Peri-Stimulus Time Histogram')

        if self.intervals_use:
            m = 0
            for j in range(0,len(self.intervals_to_use)):
                i_0, i_1 = self.intervals_to_use[m][0], self.intervals_to_use[m][1]
                plt.axvspan(i_0, i_1, alpha=0.5, color='gray')
                m+=1
        #plt.show()


    def plot_PSTH_v2(self):

        num_spikes = len(self.spike_times_to_use)

        # Define time window for averaging
        # self.analysis_begin_secs = 0  # start time
        # self.analysis_end_secs = 0.5  # end time
        dt = 0.001  # time step

        # Compute number of samples in time window
        num_samples = int((self.analysis_end_secs - self.analysis_begin_secs) / dt)

        # Initialize array to store average response for each spike
        avg_response = np.zeros(num_spikes)

        # Loop over spikes
        for i, spike_time in enumerate(self.spike_times_to_use):
            # Compute start and end times for time window
            start_time = spike_time + self.analysis_begin_secs
            end_time = min(max(self.spike_times),spike_time + self.analysis_end_secs)
            
            # Convert times to sample indices
            start_index = int(start_time / dt)
            end_index = int(end_time / dt)
            
            # Compute average response
            avg_response[i] = np.mean(self.signal_to_use[start_index:end_index])

        # Plot average response
        plt.plot(self.spike_times_to_use, avg_response)
        plt.title("PSTH")
        plt.xlabel('Time (s)')
        plt.ylabel('Average response')

        if self.intervals_use:
            m = 0
            for j in range(0,len(self.intervals_to_use)):
                i_0, i_1 = self.intervals_to_use[m][0], self.intervals_to_use[m][1]
                plt.axvspan(i_0, i_1, alpha=0.5, color='gray')
                m+=1
        #plt.show()


    def save_spikerate(self):
        #for t in range(0,int(len(self.times[-1]))):
        ''' Keeps track of the number of spikes in every 1 second interval '''

        new_second = 0 # New second?
        num_spikes = 0 # Number of spikes in the period
        self.spikerates = {} # Remember how many spikes in each period
        second_start = 1
        for spike_time in self.spike_times_to_use:
            if spike_time // second_start != new_second: # This should be 1 until the second ticks over
                self.spikerates[second_start-1] = num_spikes
                num_spikes = 0
                second_start+=1
            else:
                if self.is_in_interval(spike_time):
                    num_spikes+=1

    def spikerate_stats(self):
        '''Determine average spike rate over the whole stimulus condition (stimulated or non-stimulated)'''
        r = 0
        count_spikes = 0
  
        # Work out stdev for each bucket
        self.generate_buckets()
        for spike_time in self.spike_times_to_use:
            if self.is_in_interval(spike_time):# and self.is_in_interval(spike_time+(self.num_bins)*self.bin_width*0.001):
                count_spikes+=1
                if int(spike_time) in self.one_sec_buckets.keys():
                    self.one_sec_buckets[int(spike_time)] = self.one_sec_buckets[int(spike_time)] + 1
                else:
                    self.one_sec_buckets[int(spike_time)] = 1
        print("One second buckets - counts", self.one_sec_buckets) # This gives how many spikes in each 1sec interval
        # Base don the # of spikes in each 1sec interval we can work out the variability of this and form a confidence interval
        vals = list(self.one_sec_buckets.values())
        pop_std = np.std(vals) #np.std(vals)
        sample_std = np.std(vals,ddof=1) #np.std(vals)
        sem_ = sem(vals)

        
        spike_nums_file = exp_file.replace(".wav","_spikecounts_"+str(self.spike_threshold_sd)+"_sd"+self.filename_intervals+self.intervals_namestring+".txt") # _1st Group
        file = open(spike_nums_file,'w')
        for item in vals:
            file.write(str(item)+",")
        file.close()

        # Work out average
        total_time = 0
        for time_start, time_end in self.pairs:
            total_time+=(time_end-time_start)
        # Alternative
        #total_time = len(self.one_sec_buckets.keys()) # Each key should correspond to 1sec
        spikes_per_sec = count_spikes / total_time

        txt_file = exp_file.replace(".wav","_spikerate_"+str(self.spike_threshold_sd)+"_sd"+self.filename_intervals+self.intervals_namestring+".txt") # _1st Group
        with open(txt_file, 'w') as f:
            f.write("Count: "+str(count_spikes))
            f.write("\n")
            f.write("Total time: "+str(total_time))
            f.write("\n")
            f.write("Average per sec: "+str(spikes_per_sec)) # Spikes per second
            f.write("\n")
            f.write("Population Stdev per sec: "+str(pop_std))  
            f.write("\n")
            f.write("Sampling Stdev per sec: "+str(sample_std)) # Sampling Stdev of spikes per second
            f.write("\n")
            f.write("Sampling SEM per sec: "+str(sem_)) # Accounts for number of buckets   


    def subsequent_firing_rate(self): # Bin width in ms, num_bins will analyse (num_bins * bin_width) ms after each spike
        self.spikes_follow_rawcount = {}
        self.spikes_follow_windows_examined = {}
        self.spikes_follow_per_second = {}
        r = 0
        #print("Spike times to use", self.spike_times_to_use)


        # self.bin_width = max(1,min(25,self.bin_width,int(max(self.spike_times_to_use)/5)))
        # print("Bin width", self.bin_width)
        # self.num_bins = int(int(max(self.spike_times_to_use)) / self.bin_width) # Total self.bin_width * self.num_bins  analysed (in ms)
        # print("Num bins", self.num_bins)


        for start_spike in self.spike_times_to_use:
            if self.is_in_interval(start_spike):# and self.is_in_interval(start_spike+(self.num_bins)*self.bin_width*0.001):
                t_later_spikes = [self.spike_times_to_use[j] for j in range((r+1),len(self.spike_times_to_use)-1)]
                for bin_num in range(0,self.num_bins):
                    bin_begin_time = start_spike+(bin_num)*self.bin_width*0.001
                    bin_end_time   = start_spike+(bin_num+1)*self.bin_width*0.001

                    #if bin_begin_time > 20.5 and bin_end_time < 30.5:
                    count_spikes   = len([t for t in t_later_spikes if (t<=bin_end_time and t>bin_begin_time) ])
                    #print("Bin num %s end spike time %s count spikes %s"%(str(bin_num),str(bin_end_time),str(count_spikes)))
                    if bin_num in self.spikes_follow_rawcount.keys():
                        self.spikes_follow_rawcount[bin_num] = self.spikes_follow_rawcount[bin_num] + count_spikes
                        self.spikes_follow_windows_examined[bin_num] = self.spikes_follow_windows_examined[bin_num] + 1
                    else:
                        self.spikes_follow_rawcount[bin_num] = count_spikes
                        self.spikes_follow_windows_examined[bin_num] = 1
            r+=1


        for bin_num in range(0,self.num_bins):
            if bin_num in self.spikes_follow_windows_examined.keys():
                total_secs = self.spikes_follow_windows_examined[bin_num]*self.bin_width*0.001
                spikes_per_sec = self.spikes_follow_rawcount[bin_num]/total_secs
                self.spikes_follow_per_second[str((bin_num)*self.bin_width)+"-"+str((bin_num+1)*self.bin_width)+ "ms"] = spikes_per_sec



        dct_file = exp_file.replace(".wav","_spikerate_"+str(self.spike_threshold_sd)+"_sd"+self.filename_intervals+self.intervals_namestring+".csv") # _1st Group
        psth_file = exp_file.replace(".wav","_psth_"+str("0-"+str(int(self.bin_width*self.num_bins)))+"_"+str(self.spike_threshold_sd)+"_sd"+self.filename_intervals+self.intervals_namestring+".csv") #_1st Group
        jpg_file = exp_file.replace(".wav","_psth_"+str("0-"+str(int(self.bin_width*self.num_bins)))+"_"+str(self.spike_threshold_sd)+"_sd"+self.filename_intervals+self.intervals_namestring+".jpg") #_1st Group
        #if not self.intervals_use:
        with open(dct_file, 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in self.spikerates.items():
                writer.writerow([key, value])

        with open(psth_file, 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in self.spikes_follow_per_second.items():
                writer.writerow([key, value])

 
        plt.figure()        
        plt.margins(x=0)
        plt.bar(*zip(*self.spikes_follow_per_second.items()), align="edge", width=1.0)
        plt.title("Peri-stimulus spikes per second at various delay (bins) in ms")
        plt.xlabel('Delay (ms)')
        plt.xticks(rotation=45, fontsize=7)
        #plt.xticks(len(self.spikes_follow_per_second.keys()) + 0.5, self.spikes_follow_per_second.keys(), rotation=90, fontsize=9)
        plt.ylabel('Spikes per second')
        plt.savefig(jpg_file)
        #plt.show()







test = SpikeAnalysis(file_path=exp_file)














