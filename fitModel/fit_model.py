from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab

def fit_model_discrete_time_network_hawkes_spike_and_slab(K, T, dtmax, hypers, itter, spikes):

    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
        K=K, dt_max=dtmax,
        network_hypers=hypers)
    assert true_model.check_stability()

    true_model.generate(T=T, keep=True, print_interval=50)

    ###########################################################
    # Create a test spike and slab model
    ###########################################################

    test_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
        K=K, dt_max=dtmax, network_hypers=hypers)

    test_model.add_data(spikes)

    ###########################################################
    # Fit the test model with Gibbs sampling
    ###########################################################
    samples = []
    lps = []

    for itr in range(itter):
        print("Gibbs iteration ", itr)
        test_model.resample_model()
        lps.append(test_model.log_probability())
        samples.append(test_model.copy_sample())

    return lps, samples
