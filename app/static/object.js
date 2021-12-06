var children = []
const liveView = document.getElementById('liveView');
const ldone = document.getElementById('ld');

// Upload image with javascript
// https://stackoverflow.com/questions/22087076/how-to-make-a-simple-image-upload-using-javascript-html

window.addEventListener('load', function (){
    document.querySelector('input[type="file"]').addEventListener('change', function () {
    if (this.files && this.files[0]) {
        var img = document.querySelector('img');
           img.onload = () => {
          URL.revokeObjectURL(img.src);  // no longer needed, free memory
      }      
     img.src = URL.createObjectURL(this.files[0]); // set src to blob url

    ldone.innerText = "Model Loading...";
    
    // Remove previous labels
    for (let i = 0; i < children.length; i++) {
      liveView.removeChild(children[i]);
    }
    
    children.splice(0);

    // Make predictions
    modelPred(img);
                                                                           
    }
  });
});

// code reference is from https://codelabs.developers.google.com/codelabs/tensorflowjs-object-detection

modelPred = (img) => {

  cocoSsd.load().then(model => {
       
    // detect objects in the image.
    model.detect(img).then(predictions => {
      console.log('Predictions: ', predictions);


  
      for (let n = 0; n < predictions.length; n++) {
      // If we are over 66% sure we are sure we classified it right, draw it!
        if (predictions[n].score > 0.66) {
          const p = document.createElement('p');
          p.innerText = predictions[n].class  + ' - with ' 
            + Math.round(parseFloat(predictions[n].score) * 100) 
            + '% confidence.';
          p.style = 'margin-left: ' + predictions[n].bbox[0] + 'px; margin-top: '
            + (predictions[n].bbox[1] - 10) + 'px; width: ' 
            + (predictions[n].bbox[2] - 10) + 'px; top: 0; left: 0;';

          const highlighter = document.createElement('div');
          highlighter.setAttribute('class', 'highlighter');
          highlighter.style = 'left: ' + predictions[n].bbox[0] + 'px; top: '
            + predictions[n].bbox[1] + 'px; width: ' 
            + predictions[n].bbox[2] + 'px; height: '
            + predictions[n].bbox[3] + 'px;';

          liveView.appendChild(highlighter);
          liveView.appendChild(p);
          children.push(highlighter); 
          children.push(p);
          ldone.innerText = "Done.";
        } 
      }

  
    });
  });

}