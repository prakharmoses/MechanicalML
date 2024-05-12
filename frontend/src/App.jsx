import { BrowserRouter, Routes, Route } from 'react-router-dom';

// Importing components
import Navbar from './components/Navbar';
import SolarPanelFaultDetect from './components/SolarPanelFaultDetect';
import AboutModel from './components/AboutModel';

const App = () => {
    return (
        <BrowserRouter>
            <Navbar />
            <Routes>
                <Route path="/solar_fault_detection" element={<SolarPanelFaultDetect />} />
                <Route path="/about_model" element={<AboutModel />} />
            </Routes>
        </BrowserRouter>
    )
}

export default App;