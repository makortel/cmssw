

class TestAcceleratorServiceProducerGPUMock2: public HeterogeneousEDProducer<> {
public:
  explicit TestAcceleratorServiceProducerGPUMock2(edm::ParameterSet const& iConfig) {
  }
  ~TestAcceleratorServiceProducerGPUMock2() = default override;

private:
  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    //schedule(input1, input2);
  }

  void launchCPU() override {
    /// cpu work
  }

  void launchGPUMock(std::function<void()> callback) {
    /// GPU work
    callback();
  }

  void produceCPU(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    /// put CPU product
  }

  void produceGPUMock(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    /// put GPU product
  }
};
