const F_Sigmoid = (x) => 1 / (1 + Math.pow(Math.E, -x));
const F_ReLU = (x) => Math.max(0, x);
const F_TanH = (x) => Math.tanh(x);
const F_SoftPlus = (x) => Math.log(1 + Math.pow(Math.E, x));
const F_leakyReLU = (x) => { if(x < 0) return 0.1*x; else return x; }

module.exports.F_Sigmoid = F_Sigmoid;
module.exports.F_ReLU = F_ReLU;
module.exports.F_TanH = F_TanH;
module.exports.F_SoftPlus = F_SoftPlus;
module.exports.F_leakyReLU = F_leakyReLU;

const WEIGHT_SCALE = 5;

class Neuron {
	constructor(size, af) {
		this.weights = [];
		this.activationFunction = af;
		for(let i = 0;i < size;i ++) this.weights.push((Math.random()*2 - 1) * WEIGHT_SCALE);
		this.weights.push(0); // bias
	}
	run(input) {
		if(input.length !== this.weights.length - 1) {
			throw 'Input size mismath (neuron)';
		}

		let total = this.weights[this.weights.length - 1];
		for(let i = 0;i < input.length;i ++) {
			total += input[i] * this.weights[i];
		}
		return this.activationFunction(total);
	}
}

class Layer {
	constructor(size, prev_layer_size, af) {
		this.neurons = [];
		this.prev_layer_size = prev_layer_size;
		for(let i = 0;i < size;i ++) this.neurons.push(new Neuron(prev_layer_size, af));
	}
	run(input) {
		if(input.length !== this.prev_layer_size) {
			throw 'Input size mismath (layer)';
		}

		let output = [];
		for(let neur of this.neurons) output.push(neur.run(input));
		return output;
	}
}

class Network {
	constructor(sizes, af) {
		this.layers = [];
		this.activationFunction = af || F_ReLU; // saved as a member of Network so that copyNN can make neuron with the same activationFunction
		for(let i = 0;i < sizes.length - 1;i ++) {
			this.layers.push(new Layer(sizes[i + 1], sizes[i], this.activationFunction));
		}
	}
	run(input) {
		if(input.length !== this.layers[0].prev_layer_size) {
			throw 'Input size mismath (network)';
		}

		let output = input;
		for(let lr of this.layers) {
			output = lr.run(output);
		}

		return output;
	}
}

const copyNN = (src) => {
	// recover the layer sizes
	let arr = [src.layers[0].prev_layer_size];
	for(let lr of src.layers) {
		arr.push(lr.neurons.length);
	}

	let out = new Network(arr, src.activationFunction);

	for(let i = 0;i < out.layers.length;i ++) {
		for(let j = 0;j < out.layers[i].neurons.length;j ++) {
			for(let k = 0;k < out.layers[i].neurons[j].weights.length;k ++) {
				out.layers[i].neurons[j].weights[k] = src.layers[i].neurons[j].weights[k];
			}
		}
	}

	return out;
}

const mutate = (nn, chance) => {
	for(let lr of nn.layers) {
		for(let nr of lr.neurons) {
			for(let i = 0;i < nr.weights.length;i ++) {
				if(Math.random() < chance) nr.weights[i] = (Math.random()*2 - 1) * WEIGHT_SCALE;
			}
		}
	}
}

module.exports.Network = Network;
module.exports.copyNN  = copyNN;
module.exports.mutate  = mutate;
