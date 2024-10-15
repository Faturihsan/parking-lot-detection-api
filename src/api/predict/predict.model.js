const sharp = require('sharp');
const ort = require('onnxruntime-node');
const ffmpeg = require('fluent-ffmpeg');
const crypto = require('crypto');
// const costPrediction = require('../predict/predict.cost')
// const { storeImage } = require('./predict.service');

const yolo_classes =  ['space-empty', 'space-occupied'];

async function preprocessingImage(image) {
    const img = sharp(image);
    const md = await img.metadata();
    const [img_width,img_height] = [md.width, md.height];
    const pixels = await img.removeAlpha()
        .resize({width:640,height:640,fit:'fill'})
        .raw()
        .toBuffer();

    const red = [], green = [], blue = [];
    for (let index=0; index<pixels.length; index+=3) {
        red.push(pixels[index]/255.0);
        green.push(pixels[index+1]/255.0);
        blue.push(pixels[index+2]/255.0);
    }

    const input = [...red, ...green, ...blue];
    return [input, img_width, img_height];
} 

async function run_model(model, input) {
    try {
        // Convert input to ONNX Runtime tensor
        input = new ort.Tensor(Float32Array.from(input), [1, 3, 640, 640]);
        const outputs = await model.run({ images: input });

        const outputTensor = outputs["output0"].data;
        // const outputShape = outputTensor.dims;
    
        return { output: outputTensor };
    } catch (error) {
        console.error('Error running model inference:', error);
        throw error;
    }
}


function process_output(output, img_width, img_height) {
    let boxes = [];
    const margin = 10;
    
    for (let index = 0; index < 300; index++) {
        const class_id = output[6 * index + 5];  
        const prob = output[6 * index + 4];      

        if (prob < 0.7) {
            continue;  
        }


        const x1 = output[6 * index];
        const y1 = output[6 * index + 1];
        const x2 = output[6 * index + 2];
        const y2 = output[6 * index + 3];

        const label = yolo_classes[class_id];

        boxes.push([x1, y1, x2, y2, label, prob]);
        // console.log(boxes);
    }

    boxes = boxes.sort((box1, box2) => box2[5] - box1[5]);

    const result = [];
    const countClasses = new Array(2).fill(0); 
    while (boxes.length > 0) {
        result.push(boxes[0]);

        const classIndex = yolo_classes.indexOf(boxes[0][4]);
        countClasses[classIndex]++;

        boxes = boxes.filter(box => iou(boxes[0], box) < 0.9);
    }

    return { result, countClasses };
}




function iou(box1, box2) {
    return intersection(box1, box2) / union(box1, box2);
}

function union(box1, box2) {
    const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
    const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
    const box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1);
    const box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1);
    return box1_area + box2_area - intersection(box1, box2);
}

function intersection(box1, box2) {
    const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
    const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
    const x1 = Math.max(box1_x1, box2_x1);
    const y1 = Math.max(box1_y1, box2_y1);
    const x2 = Math.min(box1_x2, box2_x2);
    const y2 = Math.min(box1_y2, box2_y2);
    return (x2 - x1) * (y2 - y1);
}

async function predictImageSegmentation(model, images) {
    const results = [];
    const countClassesArray = [];
    
    for (const image of images) {
        try {
            const id = crypto.randomUUID();
            const [input, img_width, img_height] = await preprocessingImage(image); // img_width, img_height can be ignored
            const rawOutput = await run_model(model, input);
            const { result, countClasses } = process_output(rawOutput.output, img_width, img_height); // Pass fixed size

            const formatLabel = (label) => {
                return label.split('_').map(word => word.charAt(0) + word.slice(1)).join(' ');
            };
            const classes = result.map(item => formatLabel(item[4]));
            countClassesArray.push(countClasses);

            let svgElements = '';
            for (const [x1, y1, x2, y2, label, prob] of result) {
                const formattedLabel = formatLabel(label);

                if(label === 'space-empty'){
                    svgElements += `
                    <rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" 
                        style="fill:none;stroke:red;stroke-width:4" />`;

                    svgElements += `
                        <text x="${x1}" y="${y2 + 20}" fill="white" font-size="20" font-family="Arial" 
                            font-weight="bold">${formattedLabel}</text>`;
                } else{
                    svgElements += `
                    <rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" 
                        style="fill:none;stroke:blue;stroke-width:4" />`;

                    svgElements += `
                        <text x="${x1}" y="${y2 + 20}" fill="white" font-size="20" font-family="Arial" 
                            font-weight="bold">${formattedLabel}</text>`;
                }

            }

            const svgImage = `
                <svg width="640" height="640">
                    ${svgElements}
                </svg>
            `;

            let img = sharp(image);
            img = img.composite([{
                input: Buffer.from(svgImage),
                blend: 'over'
            }]);

            const bufferWithBoundingBoxes = await img.jpeg().toBuffer();
            const base64ImageWithBoundingBoxes = bufferWithBoundingBoxes.toString('base64');
            const filename = `processed_images/${id}.jpg`;

            console.log("filename", filename);
            results.push({ class: classes, image: base64ImageWithBoundingBoxes });

        } catch (error) {
            console.error('Error processing image:', error);
            results.push({ result: null, finalCost: null, error: error.message });
            countClassesArray.push(null);
            throw error;
        }
    }

    return { results, countClassesArray };
}



module.exports = {
    predictImageSegmentation,
};