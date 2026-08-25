// dimensions-ai 5.1.x drops the write callback when child stdin applies
// backpressure. Large Lux maps can therefore deadlock once an update exceeds
// Node's stream high-water mark. Always pass the callback to stream.write so
// it runs after the buffered chunk has been consumed.
const Module = require("module");
const originalLoad = Module._load;

Module._load = function (request, parent, isMain) {
  const loaded = originalLoad.apply(this, arguments);
  if (request === "dimensions-ai" && loaded.Agent && !loaded.Agent.__luxWritePatched) {
    const Agent = loaded.Agent;
    Agent.prototype.write = function (message, callback) {
      if (this.options.detached) {
        this.messages.push(message);
        process.nextTick(callback);
        return true;
      }
      return this.streams.in.write(message, callback);
    };
    Agent.__luxWritePatched = true;
  }
  return loaded;
};
