
let model;

const init_btn = document.querySelector('#init-btn');
const train_btn = document.querySelector('#train-btn');
const test_btn = document.querySelector('#test-btn');

const dataset = generateData(100, 0.6)
const x_train = tf.tensor2d(dataset.x_train, [dataset.x_train.length, 2])
const y_train = tf.tensor2d(dataset.y_train, [dataset.y_train.length, 1])
const x_val  = tf.tensor2d(dataset.x_val,  [dataset.x_val.length, 2])
const y_val  = tf.tensor2d(dataset.y_val,  [dataset.y_val.length, 1])


// CHART.JS VIS
var ctx = document.querySelector('#trainset')
var scatter_train = new Chart(ctx, {
    type: 'scatter',
    data: {
        datasets: [{
            data: dataset.trainPt[0],
            label: 'class 0',
            backgroundColor: 'black'
        },{
            data: dataset.trainPt[1],
            label: 'class 1',
            backgroundColor: 'red'
        }]
    },
    options: {
        responsive: false
    }
})

var ctx = document.querySelector('#valset')
var scatter_val = new Chart(ctx, {
    type: 'scatter',
    data: {
        datasets: [{
            data: dataset.valPt[0],
            label: 'class 0',
            backgroundColor: 'black'
        },{
            data: dataset.valPt[1],
            label: 'class 1',
            backgroundColor: 'red'
        },{
            type: 'bubble',
            data: [],
            label: 'new data',
            backgroundColor: '#32fa32',
            borderColor: 'green',
        }]
    },
    options: {
        responsive: false
    }
})



init_btn.addEventListener('click', ()=> {    
    var num_h = parseInt(document.querySelector('#num-hid').value)
    var lr = parseFloat(document.querySelector('#lr').value)
    
    model = tf.sequential()    
    model.add(tf.layers.dense({units: num_h,
        activation: 'sigmoid', inputShape: [2]}))
    model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}))
    const mySGD = tf.train.sgd(lr)
    model.compile({loss: 'binaryCrossentropy',
        optimizer: mySGD, metrics:['accuracy']})

    train_btn.disabled = false;
})

train_btn.addEventListener('click', async() => {    
    var msg = document.querySelector('#is-training')
    msg.classList.toggle('bg-warning')    
    msg.innerText = 'Training, please wait...';    
    
    const trainLogs = []
    const loss = document.querySelector('#loss-graph')
    const acc = document.querySelector('#acc-graph')
    var epoch = parseInt(document.querySelector('#epoch').value)
    
    const history = await model.fit(x_train, y_train, {
        epochs: epoch,
        validationData: [x_val, y_val],
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                trainLogs.push(logs)
                tfvis.show.history(loss, trainLogs, ['loss', 'val_loss'], { width: 400, height: 250 })
                tfvis.show.history(acc, trainLogs, ['acc', 'val_acc'], { width: 400, height: 250 })
            },
        },
    })
    
    let eval_train = model.evaluate(x_train, y_train)
    let eval_val = model.evaluate(x_val, y_val)
    
    msg.classList.replace('bg-warning' , 'bg-success');    
    msg.innerText = 'Training Done';
    
    let round = (num) => parseFloat(num*100).toFixed(2)
    document.querySelector('#eval-train').innerText = 'Trainset Accuracy : '+ round(eval_train[1].dataSync())+'%'
    document.querySelector('#eval-val').innerText = 'Validation Accuracy : '+ round(eval_val[1].dataSync())+'%'    
    test_btn.disabled =  false;
})

test_btn.addEventListener('click', ()=> {
    var x = parseFloat(document.querySelector('#input-x').value)
    var y = parseFloat(document.querySelector('#input-y').value)
        
    let new_dt = {'x':x,'y':y, 'r':5}
    scatter_val.data.datasets[2].data[0] = new_dt
    scatter_val.update()    
    
    let y_pred = model.predict(tf.tensor2d([[x,y]], [1,2]))
    let class_pred = 'Predicted Class: '+Math.round(y_pred.dataSync())
    document.querySelector('#class-pred').innerText = class_pred
})


function gaussianRand(a,b) {
    var rand = 0
    for (var i = 0; i < 6; i++) {
        rand += Math.random()
    }
    return (rand / 6)+(Math.random()*a+b)
}


function toArray(arr){
    var out = []
    for(var i = 0; i < arr.length; i++){
        out.push([arr[i].x,arr[i].y])
    }
    return out
}

function generateData(numPoints, frac){
    // class proportion
    let nx1 = Math.round(numPoints*frac)
    let nx2 = numPoints - nx1    
    
    // create random 2 dimension data
    let data1 = Array(nx1).fill(0).map(() =>
        {return{'x':gaussianRand(10,7),'y':gaussianRand(10,7)}})
    let class1 = Array(nx1).fill(0)
    
    let data2 = Array(nx2).fill(0).map(() =>
        {return{'x':gaussianRand(9,0),'y':gaussianRand(9,0)}})
    let class2 = Array(nx2).fill(1)
    
    // split 30% for data validation
    let nv1 = Math.round(nx1*.7)    
    let nv2 = Math.round(nx2*.7)
    
    let trainPt = [data1.slice(0,nv1), data2.slice(0,nv2)]
    let valPt = [data1.slice(nv1), data2.slice(nv2)]

    let x_train = toArray(trainPt[0].concat(trainPt[1]))
    let y_train = class1.slice(0,nv1).concat(class2.slice(0,nv2))
    let x_val  = toArray(valPt[0].concat(valPt[1]))
    let y_val  = class1.slice(nv1).concat(class2.slice(nv2))
    
    return {
        x_train, y_train, x_val, y_val, trainPt, valPt
    }
}