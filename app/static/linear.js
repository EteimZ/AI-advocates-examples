//TODO: Refactor code

let model = tf.sequential()
let init_btn = document.querySelector('#init-btn');      // initialization button
let train_btn = document.querySelector('#train-btn');   //  training button
let pred_btn = document.querySelector('#predict-btn'); //   predict button

var ctx = document.querySelector('#scatter')


const x = tf.randomUniform([100, 1]).mul(2);
const y = x.add(tf.randomNormal([100,1]).mul(3).add(4))

const arr_x = Array.from(x.dataSync());
const arr_y = Array.from(y.dataSync());


//const arr_x = [-1, -2,  0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 6]
//const arr_y = [-1, -2, -1, 1, 1, 0, 2, 3, 1, 3, 2, 4, 3, 6, 5]

let zip = (arr1, arr2) => arr1.map((x, i) => { return {'x':x, 'y':arr2[i]}})
const toy_data = zip(arr_x, arr_y)
const label = 'toy data'

//const x = tf.tensor2d(arr_x, [15, 1])
//const y = tf.tensor2d(arr_y, [15, 1])

 

// CHART.JS VIS

var scatterChart = new Chart(ctx, {
    type: 'bubble',
    data: {
        datasets: [{
            data : toy_data,
            label: label,
            backgroundColor: 'red'}]
    },
    options: {
        responsive: false
    }
})


function viewPrediction(model){    
    let t_pred = model.predict(x)
    let y_pred = t_pred.dataSync()
    let ar_pred = zip(arr_x, y_pred)
    
    scatterChart.destroy()
    scatterChart = new Chart(ctx, {
    type: 'bubble',
    data: {
        labels: arr_x,
        datasets: [
            {
                type : 'line',
                label: 'prediction',
                data : ar_pred,
                fill : false,
                borderColor: 'blue',
                pointRadius: 0
            }, {
                type : 'bubble',
                label: 'training data',
                data : toy_data,
                backgroundColor: 'red',
                borderColor: 'transparent'
            }
        ]
    },
    options: { responsive: false }
    })
}


init_btn.addEventListener( 'click' , ()=> {
    model.add(tf.layers.dense({units: 1, inputShape: [1]}))
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'})    
    
    //const mySGD = tf.train.sgd(0.1)
    //model.compile({loss: 'meanSquaredError', optimizer: mySGD})

    viewPrediction(model)
    train_btn.disabled = false;
})

train_btn.addEventListener( 'click', ()=>{    
    let msg = document.querySelector('#is-training');
    msg.classList.toggle('bg-warning');    
    msg.innerText = 'Training, please wait...';    
    
    model.fit(x, y, {epochs: 20}).then((hist) => {    
        let mse = model.evaluate(x, y)
        
        viewPrediction(model)
        msg.classList.replace('bg-warning', 'bg-success')
        msg.innerText = 'MSE: '+mse.dataSync()
        pred_btn.disabled =  false;
        
        const surface = tfvis.visor().surface({ name: 'Training History', tab: 'MSE' })    
        tfvis.show.history(surface, hist, ['loss'])        

    })
})

pred_btn.addEventListener('click' ,()=>{
    var num = parseFloat(document.querySelector('#inputValue').value)
    let y_pred = model.predict(tf.tensor2d([num], [1,1]))
    document.querySelector('#output').innerText = y_pred.dataSync()
})


// @author Eteims