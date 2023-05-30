import React from 'react'
import { BrowserRouter as Router, Route, Routes} from 'react-router-dom'

// import Home from './pages/Home/Home'

// import About from './pages/About/About'
// import Tutorials from './pages/Tutorials/Tutorials'

import './App.css'
import Posetron from './posetron/posetron'
export default function App() {
  return (
    <Router>
      <Routes>
        <Route path='/' element={<Posetron />}/>
      </Routes>
    </Router>
  )
}


