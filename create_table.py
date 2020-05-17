CREATE TABLE IF NOT EXISTS books(
    book_id INT(11) NOT NULL PRIMARY KEY,
    goodreads_book_id INT(11),
    best_book_id INT(11),
    work_id INT(11),
    books_count INT(11),
    isbn VARCHAR(50),
    isbn13 VARCHAR(50),
    authors VARCHAR(200),
    original_publication_year DATE,
    original_title VARCHAR(200),
    title VARCHAR(200),
    language_code VARCHAR(10),
    average_rating FLOAT,
    ratings_count INT(11),
    work_ratings_count INT(11),
    work_text_reviews_count INT(11),
    ratings_1 INT(11),
    ratings_2 INT(11),
    ratings_3 INT(11),
    ratings_4 INT(11),
    ratings_5 INT(11),
    image_url TEXT,
    small_image TEXT
) ENGINE=InnoDB DEFAULT charset=utf8;


CREATE TABLE IF NOT EXISTS users(
    user_id INT(11) AUTO_INCREMENT NOT NULL PRIMARY KEY,
    name VARCHAR(50),
    user_name VARCHAR(50),
    password VARCHAR(50)
) ENGINE=InnoDB DEFAULT charset=utf8;

CREATE TABLE IF NOT EXISTS ratings(
    user_id INT(11) NOT NULL,
    book_id INT(11) NOT NULL,
    rating INT(11) NOT NULL,
    CONSTRAINT pk_ratings PRIMARY KEY (user_id, book_id),
    CONSTRAINT fk_user_id FOREIGN KEY (user_id) REFERENCES users (user_id),
    CONSTRAINT fk_book_id FOREIGN KEY (book_id) REFERENCES books (book_id) 
) ENGINE=InnoDB DEFAULT charset=utf8;

LOAD DATA LOCAL INFILE '/git/Book-recommender/goodbooks-10k/books.csv' 
INTO TABLE books 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

insert into `users`
(`name`, `user_name`, `password` )
values 
('aaa', 'aaa', 'aaa' ),
('bbb', 'bbb', 'bbb' );

insert into `ratings`
(`user_id`, `book_id`, `rating` )
values 
('1', '2', '3' ),
('2', '4', '4' ),
('1', '3', '3' ),
('1', '4', '3' ),
('1', '5', '2' ),
('1', '6', '5' ),
('1', '7', '3' ),
('1', '8', '1' ),
('1', '9', '4' ),
('1', '10', '2' ),
('1', '20', '5' ),
('2', '14', '4' ),
('2', '24', '4' ),
('2', '34', '4' ),
('2', '44', '4' ),
('2', '54', '4' ),
('2', '64', '4' ),
('2', '74', '4' ),
('2', '84', '4' ),
('2', '94', '4' );

insert into `users`
(`name`, `user_name`, `password` )
values 
('ccc', 'ccc', 'ccc' ),
('ddd', 'ddd', 'ddd' ),
('eee', 'eee', 'eee' );


insert into `ratings`
(`user_id`, `book_id`, `rating` )
values 
('5', '5', '3' ),
('5', '4', '4' ),
('5', '3', '3' ),