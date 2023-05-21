# # proofs -------------------------------------------
# data <- new_matrix
# zero <- data[data[, "X1"] == 0, ][,2:4]
# one <- data[data[, "X1"] == 1, ][,2:4]
# two <- data[data[, "X1"] == 2, ][,2:4]
# three <- data[data[, "X1"] == 3, ][,2:4]
# four <- data[data[, "X1"] == 4, ][,2:4]
# five <- data[data[, "X1"] == 5, ][,2:4]
# six <- data[data[, "X1"] == 6, ][,2:4]
# seven <- data[data[, "X1"] == 7, ][,2:4]
# eight <- data[data[, "X1"] == 8, ][,2:4]
# nine <- data[data[, "X1"] == 9, ][,2:4]
# str(data)
# 
# plot(one[1:10,1:2])


# Install and load the plot3Drgl package -----------------------------
install.packages("plot3Drgl")
library(plot3Drgl)
library(rg1)
# Create an animated 3D plot using the fourth dimension
plot3d(Subject1_1_1[,1:3], col = rainbow(length(Subject1_1_1[,4])), size = 2, lwd = 7, type = "l")


# load ------------------------------------------
read_matrices <- function(directory) {
  matrices <- list()
  
  for (x in 1:10) {
    for (y in 0:9) {
      for (z in 1:10) {
        filename <- paste0("Subject", x, "-", y, "-", z, ".csv")
        filepath <- file.path(directory, filename)
        
        # Leggi la matrice dal file CSV
        matrix_data <- read.csv(filepath)
        
        # Salva la matrice nella lista
        matrices[[paste0("Subject", x, "-", y, "-", z)]] <- matrix_data
      }
    }
  }
  
  return(matrices)
}
data <- read_matrices("C:/Users/User/Desktop/SCUOLA/UNI/2. Magistrale/I ANNO/ERASMUS/UCL/examens/Data mining & decision making/projects/project 2 (21 maggio)/GestureRecognitionProject/data/Domain1_csv")
read_matrices2 <- function(directory) {
  matrices <- list()
  
  for (x in 1:10) {
    for (y in c("Cone","Cuboid","Cylinder","CylindricalPipe","Hemisphere","Pyramid","RectangularPipe","Sphere","Tetrahedron","Toroid")) {
      for (z in 1:10) {
        filename <- paste0("Subject", x, "-", y, "-", z, ".csv")
        filepath <- file.path(directory, filename)
        
        # Leggi la matrice dal file CSV
        matrix_data <- read.csv(filepath)
        
        # Salva la matrice nella lista
        matrices[[paste0("Subject", x, "-", y, "-", z)]] <- matrix_data
      }
    }
  }
  
  return(matrices)
}
data2 <- read_matrices2("C:/Users/User/Desktop/SCUOLA/UNI/2. Magistrale/I ANNO/ERASMUS/UCL/examens/Data mining & decision making/projects/project 2 (21 maggio)/GestureRecognitionProject/data/Domain4_csv")


# exploratory analysis --------------------------------
str(data)
str(data$`Subject1-0-1`)
str(data$`Subject1-0-1`$X.x.)
length(data)
length(data$`Subject1-0-1`)
length(data$`Subject1-0-1`$X.x.)
head(data$`Subject1-0-1`)


# Compute the ranges of variables -------------------------------------
variable_ranges <- lapply(data, function(matrix_data) {
  apply(matrix_data, 2, range)
})

variable_ranges2 <- lapply(data2, function(matrix_data) {
  apply(matrix_data, 2, range)
})

# Print the variable ranges
for (i in seq_along(variable_ranges)) {
  cat("Matrix", i, ":\n")
  print(variable_ranges[[i]])
  cat("\n")
}

# Print the variable ranges
for (i in seq_along(variable_ranges2)) {
  cat("Matrix", i, ":\n")
  print(variable_ranges2[[i]])
  cat("\n")
}


# plots ------------------------------------
colours()

par(mfrow = c(2, 5))
plot(data2$`Subject1-Cone-10`[,c(2,3)], size = 2, lwd = 4, xlab = "x", ylab = "y", col = "orange", type = "l")
plot(data2$`Subject2-Cone-10`[,c(2,3)], size = 2, lwd = 4, xlab = "x", ylab = "y", col = "darkgreen", type = "l")
plot(data2$`Subject3-Cone-10`[,c(2,3)], size = 2, lwd = 4, xlab = "x", ylab = "y", col = "red", type = "l")
plot(data2$`Subject4-Cone-10`[,c(2,3)], size = 2, lwd = 4, xlab = "x", ylab = "y", col = "yellow", type = "l")
plot(data2$`Subject5-Cone-10`[,c(2,3)], size = 2, lwd = 4, xlab = "x", ylab = "y", col = "grey", type = "l")
plot(data2$`Subject6-Cone-10`[,c(2,3)], size = 2, lwd = 4, xlab = "x", ylab = "y", col = "black", type = "l")
plot(data2$`Subject7-Cone-10`[,c(2,3)], size = 2, lwd = 4, xlab = "x", ylab = "y", col = "pink", type = "l")
plot(data2$`Subject8-Cone-10`[,c(2,3)], size = 2, lwd = 4, xlab = "x", ylab = "y", col = "violet", type = "l")
plot(data2$`Subject9-Cone-10`[,c(2,3)], size = 2, lwd = 4, xlab = "x", ylab = "y", col = "lightblue", type = "l")
plot(data2$`Subject10-Cone-10`[,c(2,3)], size = 2, lwd = 4, xlab = "x", ylab = "y", col = "blue", type = "l")

open3d()
plot3d(data2$`Subject1-Cone-10`[,-4], size = 2, lwd = 5, type = "l", xlab = "x", ylab = "y", zlab = "z", col = "turquoise2")
plot3d(data2$`Subject2-Cone-10`[,-4], size = 2, lwd = 4, type = "l", col = "darkgreen", add = TRUE)
plot3d(data2$`Subject3-Cone-10`[,-4], size = 2, lwd = 4, type = "l", col = "red", add = TRUE)
plot3d(data2$`Subject4-Cone-10`[,-4], size = 2, lwd = 4, type = "l", col = "blue", add = TRUE)
plot3d(data2$`Subject5-Cone-10`[,-4], size = 2, lwd = 4, type = "l", col = "yellow", add = TRUE)
plot3d(data2$`Subject6-Cone-10`[,-4], size = 2, lwd = 4, type = "l", col = "grey", add = TRUE)
plot3d(data2$`Subject7-Cone-10`[,-4], size = 2, lwd = 4, type = "l", col = "black", add = TRUE)
plot3d(data2$`Subject8-Cone-10`[,-4], size = 2, lwd = 4, type = "l", col = "pink", add = TRUE)
plot3d(data2$`Subject9-Cone-10`[,-4], size = 2, lwd = 4, type = "l", col = "violet", add = TRUE)
plot3d(data2$`Subject10-Cone-10`[,-4], size = 2, lwd = 4, type = "l", col = "lightblue", add = TRUE)


# standardisation -------------------------------------------
data$`Subject1-0-1`$X.x. <- (data$`Subject1-0-1`$X.x.- mean(data$`Subject1-0-1`$X.x.))/sqrt(var(data$`Subject1-0-1`$X.x.))
apply_command_to_matrices <- function(matrix_list) {
  for (i in seq_along(matrix_list)) {
    matrix_data <- matrix_list[[i]]
    for (j in seq_along(matrix_data)) {
      variable_name <- names(matrix_data)[j]
      matrix_data[[variable_name]] <- (matrix_data[[variable_name]] - mean(matrix_data[[variable_name]])) / sqrt(var(matrix_data[[variable_name]]))
    }
    matrix_list[[i]] <- matrix_data
  }
  return(matrix_list)
}
datastandardized <- apply_command_to_matrices(data)


# Perform PCA -------------------------------------
pca_result <- prcomp(data$`Subject10-9-1`[,1:2], scale. = TRUE)
# Extract the principal component scores
pc_scores <- pca_result$x
# Create a scatterplot of the first two principal components
plot(pc_scores[, 1], pc_scores[, 2], 
     xlab = "PC1", ylab = "PC2", 
     main = "PCA Scatterplot")
# Compute the percentage of variance explained by each principal component
variance_percentage <- round(pca_result$sdev^2 / sum(pca_result$sdev^2) * 100, 2)
# Print the percentage of variance explained by each component
cat("Percentage of Variance Explained by Each Component:\n")
cat(paste0("PC", seq_along(variance_percentage), ": ", variance_percentage, "%\n"))
# Visualize the percentage of variance explained by each component
barplot(variance_percentage, 
        xlab = "Principal Component", ylab = "Percentage of Variance Explained",
        main = "Percentage of Variance Explained by Each Component")
plot(variance_percentage)

dgitfive1 <- data.frame(rbind(data$`Subject1-5-1`,data$`Subject2-5-1`,data$`Subject3-5-1`,data$`Subject4-5-1`,
                         data$`Subject5-5-1`))
dgitfive2 <- data.frame(rbind(data$`Subject6-5-1`,data$`Subject7-5-1`,data$`Subject8-5-1`,
                         data$`Subject9-5-1`,data$`Subject10-5-1`))
res.pca <- prcomp(dgitfive2,center = TRUE, scale = TRUE )
fviz_pca_ind(res.pca,
             axes = c(1,2),
             geom = "point") + 
  geom_point(col = "tomato", size = 3, shape = 16)
plot(data$`Subject10-9-1`[,1:2])

A <- c(0.91,0.89,1,1,0.91,0.64,0.93,0.94,1,0.9)
summary(A)
sd(A)
sqrt(sum((A-mean(A))^2)/(length(A)-1))
