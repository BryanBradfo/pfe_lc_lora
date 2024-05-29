if(!require(ggplot2)) install.packages("ggplot2", dependencies=TRUE)
if(!require(reshape2)) install.packages("reshape2", dependencies=TRUE)

# Charger ggplot2 et reshape2
library(ggplot2)
library(reshape2)

# Créer le dataframe avec les données
data <- data.frame(
  n_heads = factor(rep(c(8, 12), each = 3)),
  Method = rep(c("MSA", "MLP", "MSAxMLP"), times = 2),
  Full_Accuracy = c(96.7, 97.2, 98.2, 98.9, 98.9, 98.9),
  LC_LoRA_Accuracy = c(91.4, 96.7, 94.2, 98.1, 98.1,97.8)
)

# Transformer les données en long format pour ggplot2
data_long <- melt(data, id.vars = c("n_heads", "Method"))


# Créer le graphique et la légende personnalisée
p <- ggplot(data, aes(x = n_heads, fill = Method)) +
  geom_bar(aes(y = Full_Accuracy), stat = "identity", position = position_dodge(), alpha = 0.5) +
  geom_bar(aes(y = LC_LoRA_Accuracy), stat = "identity", position = position_dodge(), alpha = 1) +
  geom_text(aes(y = Full_Accuracy, label = Full_Accuracy),
            position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
  geom_text(aes(y = LC_LoRA_Accuracy, label = LC_LoRA_Accuracy),
            position = position_dodge(width = 0.9), vjust = 1.5, size = 3) +
  scale_fill_manual(values = c("MSA" = "red", "MLP" = "green", "MSAxMLP" = "blue")) +
  labs(x = "Number of Heads", y = "Accuracy (%)",
       title = "ViT-Small on MNIST with different LoRA configurations",
       subtitle = "Transparent bars represent Full Accuracy, Colored bars represent LC+LoRA Accuracy") +
  theme_minimal() +
  guides(fill = guide_legend(override.aes = list(alpha = 1))) +
  theme(legend.position = "bottom") # Placer la légende par défaut en bas

p  # Afficher le graphique avec la légende personnalisée


# # Installer ggplot2 et reshape2 si ce n'est pas déjà fait
# if(!require(ggplot2)) install.packages("ggplot2", dependencies=TRUE)
# if(!require(reshape2)) install.packages("reshape2", dependencies=TRUE)

# # Charger ggplot2 et reshape2
# library(ggplot2)
# library(reshape2)

# # Créer le dataframe avec les données
# data <- data.frame(
#   n_heads = factor(rep(c(8, 12), each = 3)),
#   Method = rep(c("MSA", "MLP", "MSAxMLP"), times = 2),
#   Full_Accuracy = c(96.7, 97.2, 98.2, 98.9, 96.7, 89.9),
#   LC_LoRA_Accuracy = c(91.4, 96.7, 94.2, 98.1, 85.4, 80.4)
# )

# # Transformer les données en long format pour ggplot2
# data_long <- melt(data, id.vars = c("n_heads", "Method"))

# # Créer le graphique
# ggplot(data, aes(x = n_heads, fill = Method)) +
#   geom_bar(aes(y = Full_Accuracy), stat = "identity", position = position_dodge(), alpha = 0.5) +
#   geom_bar(aes(y = LC_LoRA_Accuracy), stat = "identity", position = position_dodge(), alpha = 1) +
#   geom_text(aes(y = Full_Accuracy, label = Full_Accuracy), 
#             position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
#   geom_text(aes(y = LC_LoRA_Accuracy, label = LC_LoRA_Accuracy), 
#             position = position_dodge(width = 0.9), vjust = 1.5, size = 3) +
#   scale_fill_manual(values = c("MSA" = "red", "MLP" = "green", "MSAxMLP" = "blue")) +
#   labs(x = "Number of Heads", y = "Accuracy (%)", fill = "Method",
#        title = "ViT-Small on MNIST with different LoRA configurations",
#        subtitle = "Transparent bars represent Full Accuracy, Colored bars represent LC+LoRA Accuracy") +
#   theme_minimal() +
#   guides(fill = guide_legend(override.aes = list(alpha = 1)))



# # Installer ggplot2 et reshape2 si ce n'est pas déjà fait
# if(!require(ggplot2)) install.packages("ggplot2", dependencies=TRUE)
# if(!require(reshape2)) install.packages("reshape2", dependencies=TRUE)

# # Charger ggplot2 et reshape2
# library(ggplot2)
# library(reshape2)

# # Créer le dataframe avec les données
# data <- data.frame(
#   n_heads = factor(rep(c(8, 12), each = 3)),
#   Method = rep(c("MSA", "MLP", "MSAxMLP"), times = 2),
#   Full_Accuracy = c(96.7, 97.2, 98.2, 98.9, 96.7, 89.9),
#   LC_LoRA_Accuracy = c(91.4, 96.7, 94.2, 98.1, 85.4, 80.4)
# )

# # Transformer les données en long format pour ggplot2
# data_long <- melt(data, id.vars = c("n_heads", "Method"))

# # Créer le graphique
# ggplot(data, aes(x = n_heads, fill = Method)) +
#   geom_bar(aes(y = Full_Accuracy), stat = "identity", position = position_dodge(), alpha = 0.5) +
#   geom_bar(aes(y = LC_LoRA_Accuracy), stat = "identity", position = position_dodge(), alpha = 1) +
#   geom_text(aes(y = Full_Accuracy, label = Full_Accuracy), 
#             position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
#   geom_text(aes(y = LC_LoRA_Accuracy, label = LC_LoRA_Accuracy), 
#             position = position_dodge(width = 0.9), vjust = 1.5, size = 3) +
#   scale_fill_manual(values = c("MSA" = "red", "MLP" = "green", "MSAxMLP" = "blue")) +
#   scale_alpha_manual(values = c("Full Accuracy" = 0.5, "LC+LoRA Accuracy" = 1),
#                      labels = c("Full Accuracy", "LC+LoRA Accuracy")) +
#   labs(x = "Number of Heads", y = "Accuracy (%)", fill = "Method",
#        title = "Bar plot for ViT-Small on MNIST with different configurations of LoRA") +
#   theme_minimal()



# # Installer ggplot2 et reshape2 si ce n'est pas déjà fait
# if(!require(ggplot2)) install.packages("ggplot2", dependencies=TRUE)
# if(!require(reshape2)) install.packages("reshape2", dependencies=TRUE)

# # Charger ggplot2 et reshape2
# library(ggplot2)
# library(reshape2)

# # Créer le dataframe avec les données
# data <- data.frame(
#   n_heads = factor(rep(c(8, 12), each = 3)),
#   Method = rep(c("MSA", "MLP", "MSAxMLP"), times = 2),
#   Full_Accuracy = c(96.7, 97.2, 98.2, 98.9, 96.7, 89.9),
#   LC_LoRA_Accuracy = c(91.4, 96.7, 94.2, 98.1, 85.4, 80.4)
# )

# # Transformer les données en long format pour ggplot2
# data_long <- melt(data, id.vars = c("n_heads", "Method"))

# # Créer le graphique
# ggplot(data, aes(x = n_heads, fill = Method)) +
#   geom_bar(aes(y = Full_Accuracy), stat = "identity", position = position_dodge(), alpha = 0.5) +
#   geom_bar(aes(y = LC_LoRA_Accuracy), stat = "identity", position = position_dodge(), alpha = 1) +
#   scale_fill_manual(values = c("MSA" = "red", "MLP" = "green", "MSAxMLP" = "blue")) +
#   labs(x = "Number of Heads", y = "Accuracy (%)", fill = "Method") +
#   theme_minimal()
# Installer ggplot2 et reshape2 si ce n'est pas déjà fait

# if(!require(ggplot2)) install.packages("ggplot2", dependencies=TRUE)
# if(!require(reshape2)) install.packages("reshape2", dependencies=TRUE)

# # Charger ggplot2 et reshape2
# library(ggplot2)
# library(reshape2)

# # Créer le dataframe avec les données
# data <- data.frame(
#   n_heads = factor(rep(c(8, 12), each = 3)),
#   Method = rep(c("MSA", "MLP", "MSAxMLP"), times = 2),
#   Full_Accuracy = c(96.7, 97.2, 98.2, 98.9, 96.7, 89.9),
#   LC_LoRA_Accuracy = c(91.4, 96.7, 94.2, 98.1, 85.4, 80.4)
# )

# # Transformer les données en long format pour ggplot2
# data_long <- melt(data, id.vars = c("n_heads", "Method"))

# # Créer le graphique
# ggplot(data, aes(x = n_heads, fill = Method)) +
#   geom_bar(aes(y = Full_Accuracy), stat = "identity", position = position_dodge(), alpha = 0.5) +
#   geom_bar(aes(y = LC_LoRA_Accuracy), stat = "identity", position = position_dodge(), alpha = 1) +
#   geom_text(aes(y = Full_Accuracy, label = Full_Accuracy), 
#             position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
#   geom_text(aes(y = LC_LoRA_Accuracy, label = LC_LoRA_Accuracy), 
#             position = position_dodge(width = 0.9), vjust = 1.5, size = 3) +
#   scale_fill_manual(values = c("MSA" = "red", "MLP" = "green", "MSAxMLP" = "blue")) +
#   labs(x = "Number of Heads", y = "Accuracy (%)", fill = "Method") +
#   theme_minimal()
