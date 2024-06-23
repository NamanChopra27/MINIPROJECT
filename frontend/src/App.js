import './App.css';
import LoginPage from './Pages/Login';
import Home from './Pages/Home';
import Details from './Pages/Details';
import RegisterPatient from './Pages/RegisterPatient';
import Recorder from './Pages/Recorder';
import Report from './Pages/Report';
import { Routes, Route, BrowserRouter, Navigate } from 'react-router-dom';
import { useEffect } from 'react';
import { useAuthContext } from './utils/authContext';

const App = () => {
  const { authUser } = useAuthContext();
  useEffect(()=>{
    const cleanup = () => {
      localStorage.removeItem('token');
      localStorage.removeItem('recordedAudio');
      localStorage.removeItem('pid');
    };

    window.addEventListener('beforeunload', cleanup);

    return () => {
      window.removeEventListener('beforeunload', cleanup);
    };
  },[])
  console.log('auth',authUser);
  return (
    <BrowserRouter>
      <Routes>
        <Route path='/' element={<LoginPage/>} />
        <Route path='/home'  element={authUser ? <Home /> : <LoginPage />}/>
         <Route path='/patient' element={authUser ? <Details /> : <LoginPage />}/>
       // <Route path='/registerPatient' element={authUser ? <RegisterPatient /> : <LoginPage />}/> 
        <Route path='/record' element={authUser ? <Recorder /> : <LoginPage />} />
        <Route path='/report' element={authUser ? <Report /> : <LoginPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
