// Importing styles
import '../styles/AboutModel.css';

const AboutModel = () => {
    return (
        <main className='aboutmodel-container'>
            <h1>Model Explanation: Convolutional Neural Networks</h1>
            <h2>Solar Panel Image Classification for Better Efficiency</h2>

            <div className="contents-table">
                <h3>Table of Contents</h3>
                <ul>
                    <li><a href="#Objective">Objective</a></li>
                    <li><a href="#Factorsleadingtolowerefficiency">Factors leading to lower efficiency</a></li>
                    <li><a href="#ExistingMethod">Existing Method</a></li>
                    <li><a href="#WhyMachineLearning?">Why Machine Learning?</a></li>
                    <li><a href="#DataCollection">Data Collection</a></li>
                    <li><a href="#Methodology">Methodology</a></li>
                    <li><a href="#MajorResult">Major Result</a></li>
                    <li><a href="#BonusExtras">Bonus/Extras</a></li>
                    <li><a href="#FutureEnhancements">Future Enhancements</a></li>
                    <li><a href="#References">References</a></li>
                </ul>
            </div>

            <div className="content-material" id='Objective'>
                <h3>Objective</h3>
                <p>
                    The amount of energy produced by solar panels is decreased as dust, snow, bird droppings, and other debris build up on the surface of solar panels. Keeping an eye on and cleaning solar panels is an essential duty, thus creating an ideal process to do so is critical to improving module efficiency, cutting maintenance costs, and conserving resources.<br />
                    The goal of this project is to learn and build a CNN model which can easily classify and tell which solar panels are energy efficient and which are not, if not then the reason for the lack of efficiency.
                </p>
            </div>

            <div className="content-material" id='Factorsleadingtolowerefficiency'>
                <h3>Factors leading to lower efficiency</h3>
                <ul >
                    <li>
                        Of Solar Panels:
                        <ul>
                            <li>
                                <strong>Hight Temperature:</strong> Solar panels work efficiently at moderate temperatures. At high temperatures, electrons move faster which energy can’t be fully converted to electricity. This can cause thermal damage also.
                            </li>
                            <li>
                                <strong>Low Sunlight:</strong> Lack of sunlight implies less energy to electrons so less energy generation.
                            </li>
                            <li>
                                <strong>Dirt and Debris:</strong> Dirt and debris block the surface area on which the light is falling. Due to which the efficiency of solar panel decreases drastically. It requires lost of cost and human effort for maintenance.
                            </li>
                            <li>
                                <strong>Degradation:</strong> Solar panels degrade over time and thus then are not that much efficient. Degradation rate depends on type of solar panel.
                            </li>
                        </ul>
                    </li>
                    <li>
                        Of ML model:
                        <ul>
                            <li>
                                <strong>Bad Dataset:</strong> According to the Havard Research conducted on ML models in 2001 and also in other reports too, the performance of the model largely depends on the dataset used to train it. Choosing dataset wisely is a crucial aspect of training the model.<sup><a href='#ref1'>[1]</a></sup>
                            </li>
                            <li>
                                <strong>Bad Preprocessing:</strong> Thought the model can be good but if not pre-processed properly then can’t offer it’s highest efficiency.
                            </li>
                            <li>
                                <strong>Bias and Variance:</strong> Training with less epochs can result in underfitting of the data and with more than required epochs can result in overfitting of the data. The balance should be maintained.
                            </li>
                        </ul>
                    </li>
                </ul>
            </div>

            <div className="content-material" id='ExistingMethod'>
                <h3>Existing Method</h3>
                <ul>
                    <li>
                        <strong>Manual Detection:</strong> Humans have to monitor the state of the solar panels regularly. This can be done in a fixed interval of time. Too much effort consuming.
                    </li>
                    <li>
                        <strong>Overlay Detection Technology:</strong> Overlay Detection Technology Based on Deep Learning. PV panel overlay detection technology based on deep learning is a technology that uses artificial intelligence algorithms to identify and locate foreign objects on PV panels to evaluate their impact on PV power generation efficiency.<sup><a href="#ref2">[2]</a></sup> Costly and not generally available.
                    </li>
                </ul>
            </div>

            <div className="content-material" id='WhyMachineLearning?'>
                <h3>Why Machine Learning?</h3>
                <p>
                    Since the regular checking of the solar panels for effective working is time consuming as well as demands ample amount of human effort and cost.<br />
                    Moreover, most of the faults in the solar panels can be detected just by observation, but the same observation for a solar plant flooded on the scale of acres is truly challenging task.<br />
                    Machine Learning provides a more accurate and precise way to discern the efficiency of the solar panels working. And if there is problem then it also provides the type of problem and way to resolve it.

                </p>
            </div>

            <div className="content-material" id='DataCollection'>
                <h3>Data Collection</h3>
                <p>The dataset used to train model is accessed from Kaggle datasets. The dataset contains the images of the solar panels with the following condition:</p>
                <ol>
                    <li>Working fine [efficient]</li>
                    <li>Mechanical Damage</li>
                    <li>Dusty</li>
                    <li>Unclean [Bird-drop]</li>
                    <li>Electrical Damage</li>
                    <li>Covered (by snow)</li>
                </ol>
            </div>

            <div className="content-material" id='Methodology'>
                <h3>Methodology</h3>
                <ul>
                    <li>Convolutional Neural Network has been used to detect the faults.</li>
                    <li>The training data is available in different folders having names which describe the label of the image.</li>
                    <li>Data is read using TensorFlow.</li>
                    <li>Data is then divided into training, validation and testing.</li>
                    <li>Once the data is checked by printing it.</li>
                    <li>
                        <strong>The flow of the CNN is as follows:</strong>
                        <ul>
                            <li>Convolution 1 --&gt; Convolution 2 --&gt; Pooling</li>
                            <li>Convolution 1 --&gt; Convolution 2 --&gt; Pooling</li>
                            <li>Convolution 1 --&gt; Convolution 2 --&gt; Convolution 3 --&gt; Pooling – &#10100;X 3&#10101;</li>
                            <li>3 Dense layers (fully connected)  Output layer</li>
                        </ul>
                    </li>
                    <li>
                        <strong>Upon broadly classifying, the process followed is:</strong>
                        <ul>
                            <li>Convolution Layer + ReLU</li>
                            <li>Pooling Layer</li>
                            <li>Fully-Connected Layer</li>
                            <li>SoftMax Layer</li>
                        </ul>
                    </li>
                    <li>Building and training the model with epochs = 15 using keras.</li>
                </ul>
            </div>

            <div className="content-material" id='MajorResults'>
                <h3>Major Results</h3>
                <ul>
                    <li>Accuracy of the model = 0.8725</li>
                    <li>Loss occurring = 0.4738</li>
                    <li>
                        It identified 14 out of 16 images correctly:
                        <img src="./Screenshot 2024-05-08 231535.png" alt="The Prediction accuracy" />
                        <img src="./Screenshot 2024-05-08 231619.png" alt="The Prediction accuracy" />
                    </li>
                </ul>

                <div>
                    <strong>The accuracy plot is:</strong>
                    <img src="./Screenshot 2024-05-08 232055.png" alt="Accuracy Plot" />
                </div>
                <div>
                    <strong>The loss plot is:</strong>
                    <img src="./Screenshot 2024-05-09 021458.png" alt="Loss Plot" />
                </div>
            </div>

            <div className="content-material" id='BonusExtra'>
                <h3>Bonus / Extra</h3>
                <p>A full stack development for user-friendly use is developed using React as Frontend and FastAPI as backend.</p>
                <img src="./WebFrontend.png" alt="Web Service" />
            </div>

            <div className="content-material" id='FutureEnhancements'>
                <h3>Future Enhancements</h3>
                <ul>
                    <li>More variety of dataset can be fed to detect more types of faults.</li>
                    <li>With detecting the cause and the state of the solar panel, rough figure of efficiency can also be predicted for desirable performance.</li>
                    <li>Some more models can be integrated into the service to transform it into a mechanical platform.</li>
                </ul>
            </div>

            <div className="content-material" id='References'>
                <h3>References</h3>
                <ul>
                    <li><a href="https://www.sciencedirect.com/science/article/pii/S0169260721005782" id='ref1'>[1] https://www.sciencedirect.com/science/article/pii/S0169260721005782</a></li>
                    <li><a href="https://www.mdpi.com/1996-1073/17/4/837#:~:text=Overlay%20Detection%20Technology%20Based%20on,efficiency%20%5B71%2C72%5D." id='ref2'>[2] https://www.mdpi.com/1996-1073/17/4/837#:~:text=Overlay%20Detection%20Technology%20Based%20on,efficiency%20%5B71%2C72%5D.</a></li>
                </ul>
            </div>
        </main>
    );
};

export default AboutModel;