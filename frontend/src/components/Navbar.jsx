import { Link } from 'react-router-dom'

// Importing styles
import '../styles/Navbar.css';

const Navbar = () => {
    return (
        <nav className="navbarContainer">
            <ul>
                <li><Link to='/solar_fault_detection'>Solar Panel Fault Detection</Link></li>
                <li><Link to='/about_model'>About Model</Link></li>
            </ul>
        </nav>
    )
}

export default Navbar;