import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faUser, faLock, faVenusMars, faStethoscope } from '@fortawesome/free-solid-svg-icons';
import '../Styles/Login.css';
import bgposter from '../Images/bg-poster.svg';
import axios from 'axios';
import { useAuthContext } from '../utils/authContext';

function LoginPage() {
  const navigate = useNavigate();
  const { setAuthUser } = useAuthContext();  
  const [isSignUpMode, setIsSignUpMode] = useState(false);
  const [signupError, setSignupError] = useState('');
  const [loginError, setLoginError] = useState('');
  const [chatOutput, setChatOutPut] = useState('How can I assist you today ?');
  const [chatInput,setChatInput] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    setChatOutPut('');
    try {
      const { data } = await axios.post('http://localhost:3001/api/auth/chat',{text:chatInput},{withCredentials:true});
      setChatOutPut(data.summary);
      setChatInput('');
    }catch(error){
      console.log(error);
    }
  }

  const handleChatChange = (event) => {
    setChatInput(event.target.value);
  }

  const [signUpUserDetails, setSignUpUserDetails] = useState({
		fullName: "",
		password: "",
    rePassword: "",
    gender:"",
    age:""
	});

  const [loginUserDetails, setLoginUserDetails] = useState({
		fullName: "",
		password: ""
	});

  const loginHandleChange = (event) => {
      const { name, value } = event.target;
      setLoginUserDetails(prevUserDetails => ({
        ...prevUserDetails,
        [name]:value,
      }))
    }
  
  const signupHandleChange = (event) => {
    const { name, value } = event.target;
    setSignUpUserDetails(prevUserDetails => ({
      ...prevUserDetails,
      [name]: value,
    }))
  }

  const handleSignUp = async (event) =>{
    event.preventDefault();
    try {
      console.log(loginUserDetails);
      const { data } = await axios.post('http://localhost:3001/api/auth/signup', { ...signUpUserDetails }, {
        withCredentials: true
      });
      localStorage('token',data.token);
      setAuthUser(data.data);
      navigate('/record');
    } catch (error) {
      setSignupError(error.response.data);
      setTimeout(() => {
        setSignupError('');
      }, 3000);
    }
  }

  const handleSignin = async (event) =>{
    event.preventDefault();
    try {
      console.log(loginUserDetails);
      const { data } = await axios.post('http://localhost:3001/api/auth/login', { ...loginUserDetails }, {
        withCredentials: true
      });
      localStorage.setItem('token',data.token);
      setAuthUser(data.data);
      navigate('/record');
    } catch (error) {
      setLoginError(error.response.data);
      setTimeout(() => {
        setLoginError('');
      }, 3000);
    }
  }

  const handleSignUpClick = () => {
    setIsSignUpMode(true);
  };

  const handleSignInClick = () => {
    setIsSignUpMode(false);

  };

  return (
    <div className={`loginContainer ${isSignUpMode ? 'sign-up-mode' : ''}`}>
      <div className="forms-container">
        <div className="signin-signup">
          {/* Login form */}
          <form action="#" className="sign-in-form loginForm">
            <h2 className="title">Sign in</h2>
            <div className="input-field">
              <FontAwesomeIcon icon={faUser} className='my-auto mx-auto'/>
              <input 
                className='LoginInput' 
                type="text" 
                placeholder="Full Names" 
                name='fullName'
                value={loginUserDetails.fullName}
							  onChange={loginHandleChange}
              />
            </div>
            <div className="input-field">
              <FontAwesomeIcon icon={faLock} className='my-auto mx-auto'/>
              <input 
                className='LoginInput' 
                type="password" 
                placeholder="Password" 
                name='password'
                value={loginUserDetails.password}
							  onChange={loginHandleChange}
              />
            </div>
            <button className='btn' onClick={handleSignin}>Sign In</button>
            {/* Open the modal using document.getElementById('ID').showModal() method */}
            <button className="btn" onClick={()=>document.getElementById('my_modal_5').showModal()}>Chat With Us</button>
                <dialog id="my_modal_5" className="modal modal-bottom sm:modal-middle">
                  <div className="modal-box">
                    <input placeholder='Enter your concerns' className='output w-full rounded-md p-2 no-outline' 
                      onChange={handleChatChange} 
                      name='chat' 
                      value={chatInput}/>
                    <div className='m-2'>{chatOutput}</div>
                    <div className="modal-action">
                      <form method="dialog">
                        {/* if there is a button in form, it will close the modal */}
                        <p className="btn" onClick={handleSubmit}>Submit</p>
                      </form>
                    </div>
                  </div>
                </dialog>
            <span className='error'>{loginError}</span>
          </form>

          {/* Sign-up Form */}
          <form action="#" className="sign-up-form loginForm">
            <h2 className="title">Sign up</h2>
            <div className="input-field">
              <FontAwesomeIcon icon={faUser} className='my-auto mx-auto'/>
              <input 
                className='LoginInput' 
                type="text" 
                placeholder="Full Name" 
                name='fullName'
                value={signUpUserDetails.fullName}
							  onChange={signupHandleChange}
              />
            </div>
            <div className="input-field">
              <FontAwesomeIcon icon={faLock} className='my-auto mx-auto'/>
              <input 
                className='LoginInput' 
                type="password" 
                placeholder="Password" 
                name='password'
                value={signUpUserDetails.password}
							  onChange={signupHandleChange}
              />
            </div>
            <div className="input-field">
              <FontAwesomeIcon icon={faLock} className='my-auto mx-auto'/>
              <input 
                className='LoginInput' 
                type="password" 
                placeholder="Re Enter-password" 
                name='rePassword'
                value={signUpUserDetails.rePassword}
							  onChange={signupHandleChange}
              />
            </div>
            <div className="input-field">
              <FontAwesomeIcon icon={faVenusMars} className='my-auto mx-auto'/>
              <input 
                className='LoginInput' 
                type="text" 
                placeholder="Gender" 
                name='gender'
                value={signUpUserDetails.gender}
							  onChange={signupHandleChange}
              />
            </div>
            <div className="input-field">
              <FontAwesomeIcon icon={faStethoscope} className='my-auto mx-auto'/>
              <input 
                className='LoginInput' 
                type="text" 
                placeholder="Age" 
                name='age'
                value={signUpUserDetails.age}
							  onChange={signupHandleChange}
              />
            </div>
            <button className='btn' onClick={handleSignUp}>Sign Up</button>
            <span className='error'>{signupError}</span>
          </form>
        </div>
      </div>

      <div className="panels-container">
        <div className="panel left-panel">
          <div className="content">
            <h3 className='loginh3'>Welcome back, Learner !</h3>
            <p className='loginp'>
              <p className="large">Log in to our platform to streamline your Study</p>
            <p className="large"> Get Summary To Your Lecture Anytime Anywhere In A JIFFY !</p>
            </p>
            <button className="btn transparent" onClick={handleSignUpClick}>
               SIGN UP
               </button>
          </div>
          <img src={bgposter} className="image" alt="" />
        </div>
        <div className="panel right-panel">
          <div className="content">
            <h3 className='loginh3'>Dare to learn differently?</h3>
            <p className='loginp'>
            <p className="large"> Sign up as a Learner today and embark on a journey</p>
           
           <p className="large">Together, let's shape the future of Education</p>
            </p>
            <button onClick={handleSignInClick} className="btn transparent" id="sign-in-btn">
              LOGIN
            </button>
          </div>
          <img src={bgposter} className="image" alt="" />
        </div>
      </div>
    </div>
  )}

  export default LoginPage

  