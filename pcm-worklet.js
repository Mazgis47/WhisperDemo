class Pcm16kProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.out = new Float32Array(4096);
    this.count = 0;
    this.sum = 0;
    this.n = 0;
    this.phase = 0;
    this.port.onmessage = e => {
      if (e.data === 'flush' && this.count) this.flush();
    };
  }
  flush() {
    const payload = this.out.slice(0, this.count);
    this.port.postMessage(payload.buffer, [payload.buffer]);
    this.out = new Float32Array(4096);
    this.count = 0;
  }
  process(inputs, outputs) {
    const channels = inputs[0];
    if (!channels || !channels.length) return true;
    const input = channels[0];
    // Simple box-filter downsampling from actual AudioContext rate to 16 kHz.
    // For production-quality resampling, use a band-limited resampler.
    for (let i = 0; i < input.length; i++) {
      this.sum += input[i];
      this.n++;
      this.phase += 16000 / sampleRate;
      if (this.phase >= 1) {
        this.phase -= 1;
        this.out[this.count++] = this.sum / this.n;
        this.sum = 0;
        this.n = 0;
        if (this.count === this.out.length) this.flush();
      }
    }
    // Outputs stay silent: do not play mic audio back to speakers.
    return true;
  }
}
registerProcessor('pcm16k', Pcm16kProcessor);
