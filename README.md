# Machine-Learning-Application
A ML application using streamlit of Python


Creating script for Database
-- Create the database
CREATE DATABASE daktdl;

-- Use the created database
USE daktdl;

-- Create the useraccount table
CREATE TABLE useraccount (
    UserID INT AUTO_INCREMENT PRIMARY KEY,
    UserName VARCHAR(200),
    Password VARCHAR(200),
    DateCreated DATETIME,
    LastLogin DATETIME,
    Email VARCHAR(50)
);
