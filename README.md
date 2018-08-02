# How to use
Look at test.js for example usage.  
TL;DR:  
```javascript
// import the library
const snn = require('./main.js');

// initializes random network with 4 inputs, 2 hidden layers of sized 10 and 5, and 1 output
let nn = new snn.Network([4, 10, 5, 1]); 

// define some function that measures how bad is some network
const yourLossFunc = (net) => {
	return Math.abs(net.run([1, 0, 0, 3]) - net.run([4, 2, 4.3, -12]) * 3);
}

// train the network
for(let i = 0;i < 100;i ++) {
    let copy = snn.copyNN(nn);
    snn.mutate(copy, 0.05);
    if(yourLossFunc(copy) < yourLossFunc(nn)) {
    	nn = snn.copyNN(copy);
    }
}

console.log(nn.run([1, 0, 0, 3])) // should be close to zero;
console.log(nn.run([-12, 4, 34, 12])) // can be absouletly anything
```

# TODO
* More testing
* Error checking in constructors
* Exporting / importing data (for now you can hack around with JSON.Stringify and then adding funtions)
* Implement back propagation for supervised learning (inputs with known outputs)
